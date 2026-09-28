import torch
from torch import nn
import torch.nn.functional as F
from clip_modules import clip
from model.csp import CSPInterface
from losses import ConfidenceWeightedPartialLabelSmoothing
from clip_modules.text_encoder import CustomTextEncoder


class NEP(nn.Module):
    """
    Decoupled supervision + regularization.

    Supervised losses are kept unchanged:
        loss_att, loss_obj, loss_com, loss_glb

    Current regularization contains:
        1) score ordering (softplus / hinge) for pair / attribute / object branches
        2) entropy ordering for pair / attribute / object branches

    Pair branches (com / glb) use GT score ranking on:
        pos > neg_att
        pos > neg_obj

    Attribute branch regularizer compares:
        att_pos > att_neg

    Object branch regularizer compares:
        obj_branch > obj_pos
    """

    def __init__(self, args, dataset):
        super().__init__()
        self.args = args
        self.dset = dataset

        self.clip_model = clip.load_clip(self.args)
        self.image_encoder = self.clip_model.encode_image
        self.text_encoder = CustomTextEncoder(self.clip_model)

        self.csp = CSPInterface(clip.tokenize, self.clip_model, self.dset, self.args)
        attr2idx = self.dset.attr2idx
        obj2idx = self.dset.obj2idx
        self.train_pairs = torch.tensor(
            [(attr2idx[attr], obj2idx[obj]) for attr, obj in self.dset.train_pairs],
            dtype=torch.long,
        )

        self.temp_logit = nn.Parameter(self.clip_model.logit_scale.exp().detach())
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.cross_entropy_ao = ConfidenceWeightedPartialLabelSmoothing(
            ls=getattr(args, "ls", 0.0),
            train_pairs=self.train_pairs.to(self.device),
        )

        self.reg_weight = 0.0
        self.reg_tau = 1.0
        self.logic_use_softplus = bool(getattr(args, "logic_use_softplus", True))

        self.entropy_reg_weight = 1.0
        self.entropy_tau = float(getattr(args, "entropy_tau", 1.0))
        self.entropy_eps = float(getattr(args, "entropy_eps", 1e-12))

        self.prompt_margin= float(getattr(args, "prompt_margin", 0.0))

        self.prompt_logic_on_glb = bool(getattr(args, "prompt_logic_on_glb", True))
        self.prompt_logic_on_att = bool(getattr(args, "prompt_logic_on_att", False))
        self.prompt_logic_on_obj = bool(getattr(args, "prompt_logic_on_obj", False))


        self.entropy_margin= float(getattr(args, "entropy_margin", 0.0))

        # Debug controls.
        self.debug_attr = bool(getattr(args, "debug_attr", True))
        self.debug_print_every = int(getattr(args, "debug_print_every", 20))
        self._forward_calls = 0

        self.latest_loss_items = None

    def _encode_text(self, token_ids_1d, token_emb_3d):
        text_features = self.text_encoder(
            token_ids_1d,
            token_emb_3d,
            enable_pos_emb=True,
        )
        return F.normalize(text_features, p=2, dim=-1)

    def _pairwise_order_penalty(self, higher, lower, margin):
        diff = higher - lower
        if self.logic_use_softplus:
            loss = F.softplus(margin - diff)
        else:
            loss = F.relu(margin - diff)
        return loss.mean(), diff.mean().detach()

    @staticmethod
    def _gt_score(logits, target, tau):
        probs = F.softmax(logits / tau, dim=-1)
        return probs.gather(1, target.view(-1, 1)).squeeze(1)

    def _entropy(self, logits, tau):
        probs = F.softmax(logits / tau, dim=-1)
        return -(probs * torch.log(probs.clamp_min(self.entropy_eps))).sum(dim=-1)

    def _entropy_order_penalty(self, lower_entropy_logits, higher_entropy_logits, tau, margin):
        lower_ent = self._entropy(lower_entropy_logits, tau)
        higher_ent = self._entropy(higher_entropy_logits, tau)
        loss, gap = self._pairwise_order_penalty(higher_ent, lower_ent, margin)
        return loss, gap, lower_ent.mean().detach(), higher_ent.mean().detach()

    def _pair_prompt_order_regularizer(
        self,
        pos_logits,
        neg_att_logits,
        neg_obj_logits,
        neg_both_logits,
        target,
        tau,
        prefix,
    ):
        pos_score = self._gt_score(pos_logits, target, tau)
        neg_att_score = self._gt_score(neg_att_logits, target, tau)
        neg_obj_score = self._gt_score(neg_obj_logits, target, tau)

        rank_pos_neg_att, gap_pos_neg_att = self._pairwise_order_penalty(
            pos_score, neg_att_score, self.prompt_margin
        )
        rank_pos_neg_obj, gap_pos_neg_obj = self._pairwise_order_penalty(
            pos_score, neg_obj_score, self.prompt_margin
        )
        rank_reg = (rank_pos_neg_att + rank_pos_neg_obj)

        ent_pos_neg_att, ent_gap_pos_neg_att, pos_ent_a, neg_att_ent = self._entropy_order_penalty(
            pos_logits, neg_att_logits, self.entropy_tau, self.prompt_margin
        )
        ent_pos_neg_obj, ent_gap_pos_neg_obj, pos_ent_o, neg_obj_ent = self._entropy_order_penalty(
            pos_logits, neg_obj_logits, self.entropy_tau, self.prompt_margin
        )
        ent_reg = (ent_pos_neg_att + ent_pos_neg_obj)

        stats = {
            f"{prefix}_pos_score": pos_score.mean().detach(),
            f"{prefix}_neg_att_score": neg_att_score.mean().detach(),
            f"{prefix}_neg_obj_score": neg_obj_score.mean().detach(),
            f"{prefix}_gap_pos_neg_att": gap_pos_neg_att,
            f"{prefix}_gap_pos_neg_obj": gap_pos_neg_obj,
            f"{prefix}_rank_reg": rank_reg.detach(),
            f"{prefix}_neg_att_ent": neg_att_ent.detach(),
            f"{prefix}_neg_obj_ent": neg_obj_ent.detach(),
            f"{prefix}_ent_gap_pos_neg_att": ent_gap_pos_neg_att,
            f"{prefix}_ent_gap_pos_neg_obj": ent_gap_pos_neg_obj,
            f"{prefix}_ent_reg": ent_reg.detach(),
        }
        return rank_reg, ent_reg, stats

    # def _binary_prompt_regularizer(
    #     self,
    #     higher_logits,
    #     lower_logits,
    #     target,
    #     tau,
    #     score_margin,
    #     ent_margin,
    #     prefix,
    # ):
    #     # higher_score = self._gt_score(higher_logits, target, tau)
    #     # lower_score = self._gt_score(lower_logits, target, tau)
    #     # rank_reg, gap = self._pairwise_order_penalty(higher_score, lower_score, score_margin)

    #     ent_reg, ent_gap, higher_ent, lower_ent = self._entropy_order_penalty(
    #         higher_logits, lower_logits, self.entropy_tau, ent_margin
    #     )
    #     stats = {
    #         f"{prefix}_higher_ent": higher_ent,
    #         f"{prefix}_lower_ent": lower_ent,
    #         f"{prefix}_ent_gap": ent_gap,
    #         f"{prefix}_ent_reg": ent_reg.detach(),
    #     }
    #     return ent_reg, stats

    def _maybe_debug_print(
        self,
        att,
        att_pos_log,
        att_neg_log,
        att_prompt_pos,
        att_prompt_neg,
    ):
        if not self.debug_attr:
            return
        if self.debug_print_every <= 0:
            return
        if self._forward_calls % self.debug_print_every != 0:
            return

        with torch.no_grad():
            num_attr = att_pos_log.size(-1)
            uniform_val = 1.0 / float(num_attr)

            att_pos_prob = F.softmax(att_pos_log, dim=-1)
            att_neg_prob = F.softmax(att_neg_log, dim=-1)

            pos_uniform_diff = (att_pos_prob - uniform_val).abs().mean()
            neg_uniform_diff = (att_neg_prob - uniform_val).abs().mean()
            pos_neg_prob_diff = (att_pos_prob - att_neg_prob).abs().mean()

            att_feat_norm = att.norm(dim=-1).mean()
            att_feat_std = att.std(dim=-1).mean()
            pos_log_std = att_pos_log.std(dim=-1).mean()
            neg_log_std = att_neg_log.std(dim=-1).mean()

            pos_ent = self._entropy(att_pos_log, 1.0).mean()
            neg_ent = self._entropy(att_neg_log, 1.0).mean()

            pos_cos = att_prompt_pos @ att_prompt_pos.t()
            neg_cos = att_prompt_neg @ att_prompt_neg.t()
            eye_pos = torch.eye(pos_cos.size(0), device=pos_cos.device, dtype=torch.bool)
            eye_neg = torch.eye(neg_cos.size(0), device=neg_cos.device, dtype=torch.bool)
            pos_offdiag = pos_cos[~eye_pos].mean() if pos_cos.numel() > pos_cos.size(0) else torch.tensor(0.0, device=pos_cos.device)
            neg_offdiag = neg_cos[~eye_neg].mean() if neg_cos.numel() > neg_cos.size(0) else torch.tensor(0.0, device=neg_cos.device)

            gt_pos = self._gt_score(att_pos_log, torch.arange(att_pos_log.size(0), device=att_pos_log.device) % num_attr, 1.0).mean()
            gt_neg = self._gt_score(att_neg_log, torch.arange(att_neg_log.size(0), device=att_neg_log.device) % num_attr, 1.0).mean()

            print(
                "[ATTR-DEBUG] "
                f"step={self._forward_calls} | "
                f"feat_norm={att_feat_norm.item():.4f} feat_std={att_feat_std.item():.4f} | "
                f"pos_log_std={pos_log_std.item():.6f} neg_log_std={neg_log_std.item():.6f} | "
                f"pos_uniform_diff={pos_uniform_diff.item():.6f} neg_uniform_diff={neg_uniform_diff.item():.6f} | "
                f"pos_neg_prob_diff={pos_neg_prob_diff.item():.6f} | "
                f"H_pos={pos_ent.item():.4f} H_neg={neg_ent.item():.4f} | "
                f"prompt_cos_pos={pos_offdiag.item():.4f} prompt_cos_neg={neg_offdiag.item():.4f} | "
                f"pseudo_gt_pos={gt_pos.item():.6f} pseudo_gt_neg={gt_neg.item():.6f}"
            )

    def forward(self, data):
        self._forward_calls += 1
        img, att_id, obj_id, pair_id = data[0], data[1], data[2], data[3]

        att, obj, com, glb = self.image_encoder(img)
        att_semantic_reps = F.normalize(att, p=2, dim=-1)
        obj_semantic_reps = F.normalize(obj, p=2, dim=-1)
        com_semantic_reps = F.normalize(com, p=2, dim=-1)
        glb_semantic_reps = F.normalize(glb, p=2, dim=-1)

        token_dict = self.csp.construct_token_tensors(self.train_pairs)

        com_prompt_pos = self._encode_text(self.csp.token_ids[0], token_dict["comp_pos"])
        com_prompt_neg_att = self._encode_text(self.csp.token_ids[0], token_dict["comp_neg_att"])
        com_prompt_neg_obj = self._encode_text(self.csp.token_ids[0], token_dict["comp_neg_obj"])
        com_prompt_neg_both = self._encode_text(self.csp.token_ids[0], token_dict["comp_neg_both"])

        att_prompt_pos = self._encode_text(self.csp.token_ids[1], token_dict["attr_pos"])
        att_prompt_neg = self._encode_text(self.csp.token_ids[1], token_dict["attr_neg"])

        obj_prompt_pos = self._encode_text(self.csp.token_ids[2], token_dict["obj_pos"])
        obj_prompt_neg = self._encode_text(self.csp.token_ids[2], token_dict["obj_neg"])

        com_pos_log = self.temp_logit * com_semantic_reps @ com_prompt_pos.T
        glb_pos_log = self.temp_logit * glb_semantic_reps @ com_prompt_pos.T

        com_neg_att_log = self.temp_logit * com_semantic_reps @ com_prompt_neg_att.T
        com_neg_obj_log = self.temp_logit * com_semantic_reps @ com_prompt_neg_obj.T
        com_neg_both_log = self.temp_logit * com_semantic_reps @ com_prompt_neg_both.T
        
        glb_neg_att_log = self.temp_logit * glb_semantic_reps @ com_prompt_neg_att.T
        glb_neg_obj_log = self.temp_logit * glb_semantic_reps @ com_prompt_neg_obj.T
        glb_neg_both_log = self.temp_logit * glb_semantic_reps @ com_prompt_neg_both.T

        att_pos_log = self.temp_logit * att_semantic_reps @ att_prompt_pos.T
        att_neg_log = self.temp_logit * att_semantic_reps @ att_prompt_neg.T
        obj_pos_log = self.temp_logit * obj_semantic_reps @ obj_prompt_pos.T
        obj_neg_log = self.temp_logit * obj_semantic_reps @ obj_prompt_neg.T

        self._maybe_debug_print(att, att_pos_log, att_neg_log, att_prompt_pos, att_prompt_neg)

        att_branch = att_pos_log - att_neg_log
        obj_branch = obj_pos_log - obj_neg_log

        att_loss = F.cross_entropy(att_branch, att_id)
        obj_loss = F.cross_entropy(obj_branch, obj_id)
        com_loss = self.cross_entropy_ao(com_pos_log, pair_id)
        glb_loss = self.cross_entropy_ao(glb_pos_log, pair_id)

        total_loss = (
            self.args.att_loss_weight * att_loss
            + self.args.obj_loss_weight * obj_loss
            + self.args.com_loss_weight * com_loss
            + self.args.glb_loss_weight * glb_loss
        )

        rank_terms = []
        ent_terms = []
        reg_stats = {}

        com_rank_reg, com_ent_reg, com_reg_stats = self._pair_prompt_order_regularizer(
            com_pos_log,
            com_neg_att_log,
            com_neg_obj_log,
            com_neg_both_log,
            pair_id,
            self.reg_tau,
            "com",
        )
        rank_terms.append(com_rank_reg)
        ent_terms.append(com_ent_reg)
        reg_stats.update(com_reg_stats)

        glb_rank_reg, glb_ent_reg, glb_reg_stats = self._pair_prompt_order_regularizer(
            glb_pos_log,
            glb_neg_att_log,
            glb_neg_obj_log,
            glb_neg_both_log,
            pair_id,
            self.reg_tau,
            "glb",
        )
        rank_terms.append(glb_rank_reg)
        ent_terms.append(glb_ent_reg)
        reg_stats.update(glb_reg_stats)

        # att_ent_reg, att_reg_stats = self._binary_prompt_regularizer(
        #     att_pos_log,
        #     att_neg_log,
        #     att_id,
        #     self.reg_tau,
        #     self.prompt_margin,
        #     self.entropy_margin,
        #     "att",
        # )
        # # rank_terms.append(att_rank_reg)
        # ent_terms.append(att_ent_reg)
        # reg_stats.update(att_reg_stats)

        # obj_ent_reg, obj_reg_stats = self._binary_prompt_regularizer(
        #     obj_branch,
        #     obj_pos_log,
        #     obj_id,
        #     self.reg_tau,
        #     self.prompt_margin,
        #     self.entropy_margin,
        #     "obj",
        # )
        # # rank_terms.append(obj_rank_reg)
        # ent_terms.append(obj_ent_reg)
        # reg_stats.update(obj_reg_stats)

        zero = torch.zeros((), device=total_loss.device, dtype=total_loss.dtype)
        rank_total = self.reg_weight * sum(rank_terms) if len(rank_terms) > 0 else zero
        ent_total = self.entropy_reg_weight * sum(ent_terms) if len(ent_terms) > 0 else zero

        total_loss = total_loss + rank_total + ent_total

        zero_val = torch.tensor(0.0, device=total_loss.device)
        self.latest_loss_items = {
            "total": float(total_loss.detach().item()),
            "att": float(att_loss.detach().item()),
            "obj": float(obj_loss.detach().item()),
            "com": float(com_loss.detach().item()),
            "glb": float(glb_loss.detach().item()),
            # "reg": float(reg_total.detach().item()),
            # "reg_rank": float(reg_rank.detach().item()),
            # "reg_ent": float(reg_ent.detach().item()),

            "com_pos": float(reg_stats.get("com_pos_score", zero_val).item()),
            "com_neg_att": float(reg_stats.get("com_neg_att_score", zero_val).item()),
            "com_neg_obj": float(reg_stats.get("com_neg_obj_score", zero_val).item()),
            "com_gap_att": float(reg_stats.get("com_gap_pos_neg_att", zero_val).item()),
            "com_gap_obj": float(reg_stats.get("com_gap_pos_neg_obj", zero_val).item()),
            "com_H_pos": float(reg_stats.get("com_pos_ent", zero_val).item()),
            "com_H_neg_att": float(reg_stats.get("com_neg_att_ent", zero_val).item()),
            "com_H_neg_obj": float(reg_stats.get("com_neg_obj_ent", zero_val).item()),
            "com_Hgap_att": float(reg_stats.get("com_ent_gap_pos_neg_att", zero_val).item()),
            "com_Hgap_obj": float(reg_stats.get("com_ent_gap_pos_neg_obj", zero_val).item()),

            "glb_pos": float(reg_stats.get("glb_pos_score", zero_val).item()),
            "glb_neg_att": float(reg_stats.get("glb_neg_att_score", zero_val).item()),
            "glb_neg_obj": float(reg_stats.get("glb_neg_obj_score", zero_val).item()),
            "glb_gap_att": float(reg_stats.get("glb_gap_pos_neg_att", zero_val).item()),
            "glb_gap_obj": float(reg_stats.get("glb_gap_pos_neg_obj", zero_val).item()),
            "glb_H_pos": float(reg_stats.get("glb_pos_ent", zero_val).item()),
            "glb_H_neg_att": float(reg_stats.get("glb_neg_att_ent", zero_val).item()),
            "glb_H_neg_obj": float(reg_stats.get("glb_neg_obj_ent", zero_val).item()),
            "glb_Hgap_att": float(reg_stats.get("glb_ent_gap_pos_neg_att", zero_val).item()),
            "glb_Hgap_obj": float(reg_stats.get("glb_ent_gap_pos_neg_obj", zero_val).item()),

            "att_pos": float(reg_stats.get("att_higher_score", zero_val).item()),
            "att_neg": float(reg_stats.get("att_lower_score", zero_val).item()),
            "att_gap": float(reg_stats.get("att_gap", zero_val).item()),
            "att_H_pos": float(reg_stats.get("att_higher_ent", zero_val).item()),
            "att_H_neg": float(reg_stats.get("att_lower_ent", zero_val).item()),
            "att_Hgap": float(reg_stats.get("att_ent_gap", zero_val).item()),

            "obj_main": float(reg_stats.get("obj_higher_score", zero_val).item()),
            "obj_pos": float(reg_stats.get("obj_lower_score", zero_val).item()),
            "obj_gap": float(reg_stats.get("obj_gap", zero_val).item()),
            "obj_H_main": float(reg_stats.get("obj_higher_ent", zero_val).item()),
            "obj_H_pos": float(reg_stats.get("obj_lower_ent", zero_val).item()),
            "obj_Hgap": float(reg_stats.get("obj_ent_gap", zero_val).item()),
        }
        return total_loss
