import torch
from torch import nn


class CSPInterface(nn.Module):
    def __init__(self, tokenize, clip_model, dset, args):
        super().__init__()
        self.tokenize = tokenize
        self.config = args
        self.dset = dset
        self.clip_model = clip_model
        self.attr_dropout = nn.Dropout(getattr(args, "attr_dropout", 0.0))
        self.neg_attr_dropout = nn.Dropout(getattr(args, "attr_dropout", 0.0))

        # Debug flags for prompt-collapse diagnosis
        self.debug_attr = bool(getattr(args, "debug_attr", True))
        self.debug_print_every = max(1, int(getattr(args, "debug_print_every", 20)))
        self._debug_counter = 0

        (
            self.token_ids,
            soft_att_obj,
            neg_soft_att_obj,
            com_ctx_vectors,
            att_ctx_vectors,
            obj_ctx_vectors,
        ) = self.construct_soft_prompt()
        self.com_ctx_vectors = nn.Parameter(com_ctx_vectors)
        self.att_ctx_vectors = nn.Parameter(att_ctx_vectors)
        self.obj_ctx_vectors = nn.Parameter(obj_ctx_vectors)
        self.soft_att_obj = nn.Parameter(soft_att_obj)
        self.neg_soft_att_obj = nn.Parameter(neg_soft_att_obj)

    @staticmethod
    def _pairwise_cos_mean(x: torch.Tensor) -> torch.Tensor:
        if x.size(0) <= 1:
            return x.new_tensor(1.0)
        x = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        sim = x @ x.t()
        eye = torch.eye(sim.size(0), device=sim.device, dtype=torch.bool)
        return sim[~eye].mean()

    def get_text_embedding(self, texts):
        """
        Get CLIP text embeddings for a list of texts.

        Args:
            texts: List[str], e.g. ["no red", "no blue", "no green"]

        Returns:
            embeddings: [len(texts), embed_dim]
        """
        device = next(self.clip_model.parameters()).device

        tokenized = self.tokenize(
            texts,
            context_length=self.config.context_length,
        ).to(device)

        with torch.no_grad():
            token_embedding = self.clip_model.token_embedding(tokenized)

        embeddings = []

        for idx, rep in enumerate(token_embedding):
            # Find EOS position.
            eos_idx = tokenized[idx].argmax()

            # Average the semantic tokens, excluding SOS and EOS.
            if eos_idx > 1:
                embedding = rep[1:eos_idx, :].mean(dim=0)
            else:
                embedding = rep.mean(dim=0)

            embeddings.append(embedding)

        return torch.stack(embeddings, dim=0)


    def _maybe_debug_prompt_params(self):
        if not self.debug_attr:
            return
        if self._debug_counter % self.debug_print_every != 0:
            return
        with torch.no_grad():
            attr_pos = self.soft_att_obj[: self.num_att]
            attr_neg = self.neg_soft_att_obj[: self.num_att]
            msg = (
                f"[CSP-DEBUG:param] step={self._debug_counter} | "
                f"soft_attr_std={attr_pos.std(dim=0).mean().item():.6f} "
                f"neg_attr_std={attr_neg.std(dim=0).mean().item():.6f} | "
                f"soft_attr_cos={self._pairwise_cos_mean(attr_pos).item():.4f} "
                f"neg_attr_cos={self._pairwise_cos_mean(attr_neg).item():.4f}"
            )
            print(msg)

    def construct_soft_prompt(self):
        classes = [cla.replace(".", " ").lower() for cla in self.dset.objs]
        attributes = [attr.replace(".", " ").lower() for attr in self.dset.attrs]

        self.num_att = len(attributes)
        self.num_cls = len(classes)

        token_ids = self.tokenize(
            self.config.prompt_template,
            context_length=self.config.context_length,
        ).cuda()

        tokenized = torch.cat(
            [
                self.tokenize(tok, context_length=self.config.context_length)
                for tok in attributes + classes
            ]
        )
        orig_token_embedding = self.clip_model.token_embedding(tokenized.cuda())
        soft_att_obj = torch.zeros(
            (len(attributes) + len(classes), orig_token_embedding.size(-1)),
            device=orig_token_embedding.device,
            dtype=orig_token_embedding.dtype,
        )
        for idx, rep in enumerate(orig_token_embedding):
            eos_idx = tokenized[idx].argmax()
            soft_att_obj[idx, :] = torch.mean(rep[1:eos_idx, :], axis=0)

        ctx_init = self.config.ctx_init
        assert isinstance(ctx_init, list)
        n_ctx = [len(ctx.split()) for ctx in ctx_init]
        prompt = self.tokenize(
            ctx_init,
            context_length=self.config.context_length,
        ).cuda()
        with torch.no_grad():
            embedding = self.clip_model.token_embedding(prompt)

        comp_ctx_vectors = embedding[0, 1 : 1 + n_ctx[0], :].to(self.clip_model.dtype)
        attr_ctx_vectors = embedding[1, 1 : 1 + n_ctx[1], :].to(self.clip_model.dtype)
        obj_ctx_vectors = embedding[2, 1 : 1 + n_ctx[2], :].to(self.clip_model.dtype)

        Na = len(attributes)
        No = len(classes)

        # ---------------------------------------------------------
        # Positive attribute/object embeddings
        # ---------------------------------------------------------
        attr_pos = soft_att_obj[:Na, :]
        obj_pos = soft_att_obj[Na:, :]

        # if Na > 1:
        #     neg_topk = max(1, int(getattr(self.config, "neg_topk", 3)))
        #     k = min(neg_topk, Na - 1)
        #     attr_pos_n = attr_pos / attr_pos.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        #     sim = attr_pos_n @ attr_pos_n.t()
        #     sim.fill_diagonal_(-1e4)
        #     topk_idx = sim.topk(k=k, dim=1).indices
        #     attr_neg = attr_pos[topk_idx].mean(dim=1)
        # else:
        #     attr_neg = attr_pos.clone()

        # ---------------------------------------------------------
        # Negative attribute embeddings
        # Initialize with linguistic prompt: "no [attribute]"
        # ---------------------------------------------------------
        if Na > 1:
            attr_sum = attr_pos.sum(dim=0, keepdim=True)
            attr_neg = (attr_sum - attr_pos) / (No - 1)
        else:
            attr_neg = attr_pos.clone()
        # attr_neg = self.get_text_embedding(
        #     [f"no {attr}" for attr in attributes]
        # ).to(dtype=soft_att_obj.dtype)

        # ---------------------------------------------------------
        # Negative object embeddings
        # Mean of all alternative objects
        # ---------------------------------------------------------
        if No > 1:
            obj_sum = obj_pos.sum(dim=0, keepdim=True)
            obj_neg = (obj_sum - obj_pos) / (No - 1)
        else:
            obj_neg = obj_pos.clone()
        # obj_neg = self.get_text_embedding(
        #     [f"no {obj}" for obj in classes]
        # ).to(dtype=soft_att_obj.dtype)
        
        obj_neg = obj_neg.to(dtype=soft_att_obj.dtype)
        # ---------------------------------------------------------
        # Combine negative attribute/object embeddings
        # ---------------------------------------------------------
        neg_soft_att_obj = torch.cat(
            [attr_neg, obj_neg],
            dim=0
        ).to(dtype=soft_att_obj.dtype)

        neg_soft_att_obj = torch.cat([attr_neg, obj_neg], dim=0).to(dtype=soft_att_obj.dtype)

        if self.debug_attr:
            with torch.no_grad():
                pos_cos = self._pairwise_cos_mean(attr_pos).item()
                neg_cos = self._pairwise_cos_mean(attr_neg).item()
                pos_std = attr_pos.std(dim=0).mean().item()
                neg_std = attr_neg.std(dim=0).mean().item()
                print(
                    "[CSP-DEBUG:init] "
                    f"attr_pos_std={pos_std:.6f} attr_neg_std={neg_std:.6f} | "
                    f"attr_pos_cos={pos_cos:.4f} attr_neg_cos={neg_cos:.4f} | "
                    f"neg_topk={max(1, int(getattr(self.config, 'neg_topk', 3)))}"
                )

        return (
            token_ids,
            soft_att_obj,
            neg_soft_att_obj,
            comp_ctx_vectors,
            attr_ctx_vectors,
            obj_ctx_vectors,
        )

    def construct_token_tensors(self, pair_idx):
        device = self.clip_model.token_embedding.weight.device
        dtype = self.clip_model.dtype

        pair_idx = pair_idx.to(device)
        attr_idx = pair_idx[:, 0].long()
        obj_idx = pair_idx[:, 1].long()
        B = pair_idx.size(0)

        comp_ids = self.token_ids[0].unsqueeze(0).repeat(B, 1).to(device)
        attr_ids = self.token_ids[1].unsqueeze(0).repeat(self.num_att, 1).to(device)
        obj_ids = self.token_ids[2].unsqueeze(0).repeat(self.num_cls, 1).to(device)

        comp_base = self.clip_model.token_embedding(comp_ids).type(dtype)
        attr_base = self.clip_model.token_embedding(attr_ids).type(dtype)
        obj_base = self.clip_model.token_embedding(obj_ids).type(dtype)

        comp_pos = comp_base.clone()
        comp_neg_att = comp_base.clone()
        comp_neg_obj = comp_base.clone()
        comp_neg_both = comp_base.clone()
        attr_pos = attr_base.clone()
        attr_neg = attr_base.clone()
        obj_pos = obj_base.clone()
        obj_neg = obj_base.clone()

        eos_comp = int(self.token_ids[0].argmax())
        eos_attr = int(self.token_ids[1].argmax())
        eos_obj = int(self.token_ids[2].argmax())

        soft_att_obj_pos = self.attr_dropout(self.soft_att_obj)
        soft_att_obj_neg = self.neg_attr_dropout(self.neg_soft_att_obj)

        comp_pos[:, eos_comp - 2, :] = soft_att_obj_pos[attr_idx].type(dtype)
        comp_pos[:, eos_comp - 1, :] = soft_att_obj_pos[obj_idx + self.num_att].type(dtype)
        comp_pos[:, 1 : 1 + len(self.com_ctx_vectors), :] = self.com_ctx_vectors.type(dtype)

        comp_neg_att[:, eos_comp - 2, :] = soft_att_obj_neg[attr_idx].type(dtype)
        comp_neg_att[:, eos_comp - 1, :] = soft_att_obj_pos[obj_idx + self.num_att].type(dtype)
        comp_neg_att[:, 1 : 1 + len(self.com_ctx_vectors), :] = self.com_ctx_vectors.type(dtype)

        comp_neg_obj[:, eos_comp - 2, :] = soft_att_obj_pos[attr_idx].type(dtype)
        comp_neg_obj[:, eos_comp - 1, :] = soft_att_obj_neg[obj_idx + self.num_att].type(dtype)
        comp_neg_obj[:, 1 : 1 + len(self.com_ctx_vectors), :] = self.com_ctx_vectors.type(dtype)

        comp_neg_both[:, eos_comp - 2, :] = soft_att_obj_neg[attr_idx].type(dtype)
        comp_neg_both[:, eos_comp - 1, :] = soft_att_obj_neg[obj_idx + self.num_att].type(dtype)
        comp_neg_both[:, 1 : 1 + len(self.com_ctx_vectors), :] = self.com_ctx_vectors.type(dtype)

        attr_pos[:, eos_attr - 1, :] = soft_att_obj_pos[: self.num_att].type(dtype)
        attr_pos[:, 1 : 1 + len(self.att_ctx_vectors), :] = self.att_ctx_vectors.type(dtype)
    
        attr_neg[:, eos_attr - 1, :] = soft_att_obj_neg[: self.num_att].type(dtype)
        attr_neg[:, 1 : 1 + len(self.att_ctx_vectors), :] = self.att_ctx_vectors.type(dtype)

        obj_pos[:, eos_obj - 1, :] = soft_att_obj_pos[self.num_att :].type(dtype)
        obj_pos[:, 1 : 1 + len(self.obj_ctx_vectors), :] = self.obj_ctx_vectors.type(dtype)

        obj_neg[:, eos_obj - 1, :] = soft_att_obj_neg[self.num_att :].type(dtype)
        obj_neg[:, 1 : 1 + len(self.obj_ctx_vectors), :] = self.obj_ctx_vectors.type(dtype)

        self._debug_counter += 1
        self._maybe_debug_prompt_params()
        if self.debug_attr and self._debug_counter % self.debug_print_every == 0:
            with torch.no_grad():
                attr_pos_tokens = attr_pos[:, eos_attr - 1, :]
                attr_neg_tokens = attr_neg[:, eos_attr - 1, :]
                print(
                    "[CSP-DEBUG:tensor] "
                    f"step={self._debug_counter} | "
                    f"attr_pos_token_std={attr_pos_tokens.std(dim=0).mean().item():.6f} "
                    f"attr_neg_token_std={attr_neg_tokens.std(dim=0).mean().item():.6f} | "
                    f"attr_pos_token_cos={self._pairwise_cos_mean(attr_pos_tokens).item():.4f} "
                    f"attr_neg_token_cos={self._pairwise_cos_mean(attr_neg_tokens).item():.4f}"
                )

        return {
            "comp_pos": comp_pos,
            "comp_neg_att": comp_neg_att,
            "comp_neg_obj": comp_neg_obj,
            "comp_neg_both": comp_neg_both,
            "attr_pos": attr_pos,
            "attr_neg": attr_neg,
            "obj_pos": obj_pos,
            "obj_neg": obj_neg,
        }
