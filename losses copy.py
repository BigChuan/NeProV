import torch
import torch.nn as nn
import torch.nn.functional as F


class RelationAwarePairLabelSmoothing(nn.Module):
    """
    Label smoothing over *train pairs only*, with relation-aware mass allocation.

    Relation table codes:
        0: self / GT pair
        1: same attribute, different object
        2: same object, different attribute
        3: different attribute and different object
    """

    REL_SELF = 0
    REL_SAME_ATTR = 1
    REL_SAME_OBJ = 2
    REL_OTHER = 3

    def __init__(
        self,
        ls,
        train_pairs,
        same_attr_weight=1.0,
        same_obj_weight=1.25,
        other_weight=0.25,
        eps=1e-12,
    ):
        super().__init__()
        train_pairs = train_pairs.long()
        self.ls = float(ls)
        self.same_attr_weight = float(same_attr_weight)
        self.same_obj_weight = float(same_obj_weight)
        self.other_weight = float(other_weight)
        self.eps = float(eps)

        relation_table, same_attr_mask, same_obj_mask, other_mask = self._build_relation_table(train_pairs)
        self.register_buffer("train_pairs", train_pairs)
        self.register_buffer("relation_table", relation_table)
        self.register_buffer("same_attr_mask", same_attr_mask)
        self.register_buffer("same_obj_mask", same_obj_mask)
        self.register_buffer("other_mask", other_mask)

    @classmethod
    def _build_relation_table(cls, train_pairs):
        attr = train_pairs[:, 0:1]
        obj = train_pairs[:, 1:2]
        same_attr = attr.eq(attr.t())
        same_obj = obj.eq(obj.t())
        eye = torch.eye(train_pairs.size(0), device=train_pairs.device, dtype=torch.bool)

        same_attr_mask = same_attr & (~same_obj)
        same_obj_mask = same_obj & (~same_attr)
        other_mask = (~eye) & (~same_attr_mask) & (~same_obj_mask)

        relation_table = torch.full(
            (train_pairs.size(0), train_pairs.size(0)),
            cls.REL_OTHER,
            device=train_pairs.device,
            dtype=torch.long,
        )
        relation_table[same_attr_mask] = cls.REL_SAME_ATTR
        relation_table[same_obj_mask] = cls.REL_SAME_OBJ
        relation_table[eye] = cls.REL_SELF
        return relation_table, same_attr_mask, same_obj_mask, other_mask

    def build_soft_targets(self, target):
        num_classes = self.train_pairs.size(0)
        target = target.long()
        soft = torch.zeros(target.size(0), num_classes, device=target.device)
        soft.scatter_(1, target.unsqueeze(1), 1.0)

        if self.ls <= 0:
            return soft

        same_attr = self.same_attr_mask.index_select(0, target).float()
        same_obj = self.same_obj_mask.index_select(0, target).float()
        other = self.other_mask.index_select(0, target).float()

        weights = (
            self.same_attr_weight * same_attr
            + self.same_obj_weight * same_obj
            + self.other_weight * other
        )
        denom = weights.sum(dim=1, keepdim=True)
        has_support = denom.squeeze(1) > self.eps

        soft = soft * (1.0 - self.ls)
        if has_support.any():
            soft[has_support] = soft[has_support] + self.ls * weights[has_support] / denom[has_support]
        if (~has_support).any():
            soft[~has_support].zero_()
            soft[~has_support].scatter_(1, target[~has_support].unsqueeze(1), 1.0)
        return soft

    def forward(self, logits, target):
        soft_targets = self.build_soft_targets(target)
        log_prob = F.log_softmax(logits, dim=-1)
        return -(soft_targets * log_prob).sum(dim=-1).mean()


class PartialLabelSmoothing(nn.CrossEntropyLoss):
    def __init__(self, ls: float = 0, train_pairs=None, **kwargs):
        super().__init__(**kwargs)
        """
        if smoothing == 0, it's one-hot method
        if 0 < smoothing < 1, it's smooth method
        """
        assert 0 <= ls < 1
        self.ls = ls
        self.comp_pairs = train_pairs  # (K, 2), initialized in train.py

    
    def smooth_one_hot(self, true_labels: torch.Tensor, smooth_mask: torch.Tensor):
        """ partially smooth the labels by mask
            true_labels: (B,), values range from 0 to K-1
            smooth_mask: (B, K)
        """
        confidence = 1.0 - self.ls
        total_class = smooth_mask.size(-1)
        with torch.no_grad():
            nums_smothclass = torch.sum(smooth_mask, dim=-1, keepdim=True)  # (B, 1)
            batch_ls = torch.where(nums_smothclass > 1, self.ls, 0.0)  # if only one attribute, no need to smooth
            true_dist = batch_ls / (nums_smothclass - 1 + 1e-8).repeat(1, total_class)  # (B, K)
            true_dist = true_dist * smooth_mask
            true_dist.scatter_(1, true_labels.data.unsqueeze(1), confidence)
        return true_dist
    
    
    # def forward_(self, input: torch.Tensor, target: torch.Tensor):
    #     """ input: (B, K), the logits
    #         target: (B,), the target labels ranging from 0 to K-1
    #     """
    #     batch_size, total_class = input.size()
        
    #     # compute the smoothing mask
    #     target_attr = self.comp_pairs[target, 0].unsqueeze(1).repeat(1, total_class)  # (B, K)
    #     all_attr = self.comp_pairs[:, 0].unsqueeze(0).repeat(batch_size, 1)  # (B, K)
    #     smooth_mask = torch.where(all_attr == target_attr, 1.0, 0.0)

    #     # Convert labels to distributions
    #     smooth_labels = self.smooth_one_hot(target, smooth_mask)
    #     preds = input.log_softmax(dim=-1)
    #     return torch.mean(torch.sum(-smooth_labels * preds, dim=-1))
    

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        """
        input:  (B, K) logits
        target: (B,)   labels in [0, K-1]
        """
        batch_size, total_class = input.size()

        # ----- attr mask: same attribute -----
        target_attr = self.comp_pairs[target, 0].unsqueeze(1).repeat(1, total_class)  # (B,K)
        all_attr    = self.comp_pairs[:, 0].unsqueeze(0).repeat(batch_size, 1)        # (B,K)
        mask_attr   = (all_attr == target_attr)                                       # bool (B,K)

        # ----- obj mask: same object -----
        target_obj = self.comp_pairs[target, 1].unsqueeze(1).repeat(1, total_class)   # (B,K)
        all_obj    = self.comp_pairs[:, 1].unsqueeze(0).repeat(batch_size, 1)         # (B,K)
        mask_obj   = (all_obj == target_obj)                                          # bool (B,K)

        # ----- union: same attr OR same obj -----
        smooth_mask = (mask_attr | mask_obj).float()                                  # (B,K) in {0,1}

        # Convert labels to distributions
        smooth_labels = self.smooth_one_hot(target, smooth_mask)
        preds = input.log_softmax(dim=-1)
        return torch.mean(torch.sum(-smooth_labels * preds, dim=-1))
    
    
    
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# class PartialLabelSmoothing(nn.CrossEntropyLoss):
#     def __init__(
#         self,
#         ls: float = 0,
#         train_pairs=None,
#         obj_ratio: float = 0.7,
#         attr_ratio: float = 0.3,
#         **kwargs
#     ):
#         super().__init__(**kwargs)

#         assert 0 <= ls < 1
#         assert abs(obj_ratio + attr_ratio - 1.0) < 1e-6

#         self.ls = ls
#         self.comp_pairs = train_pairs

#         self.obj_ratio = obj_ratio
#         self.attr_ratio = attr_ratio

#     def forward(self, input: torch.Tensor, target: torch.Tensor):
#         """
#         input:  (B, K) logits
#         target: (B,) labels

#         Label distribution:
#             GT          : 1 - ls
#             Same object : ls * obj_ratio
#             Same attr   : ls * attr_ratio

#         The smoothing mass is uniformly distributed within
#         each relation group.
#         """

#         batch_size, total_class = input.size()

#         # --------------------------------------------------
#         # 1. Relation masks
#         # --------------------------------------------------
#         target_attr = self.comp_pairs[target, 0].unsqueeze(1)
#         target_obj = self.comp_pairs[target, 1].unsqueeze(1)

#         all_attr = self.comp_pairs[:, 0].unsqueeze(0)
#         all_obj = self.comp_pairs[:, 1].unsqueeze(0)

#         same_attr = (all_attr == target_attr)
#         same_obj = (all_obj == target_obj)

#         # Exclude GT itself from smoothing
#         gt_mask = torch.zeros_like(same_attr)
#         gt_mask.scatter_(1, target.unsqueeze(1), True)

#         same_attr = same_attr & (~gt_mask)
#         same_obj = same_obj & (~gt_mask)

#         # --------------------------------------------------
#         # 2. Count candidates
#         # --------------------------------------------------
#         num_attr = same_attr.sum(dim=1, keepdim=True).float()
#         num_obj = same_obj.sum(dim=1, keepdim=True).float()

#         # --------------------------------------------------
#         # 3. Allocate smoothing mass
#         # --------------------------------------------------
#         attr_mass = self.ls * self.attr_ratio
#         obj_mass = self.ls * self.obj_ratio

#         # --------------------------------------------------
#         # 4. Uniformly distribute within each group
#         # --------------------------------------------------
#         attr_prob = torch.where(
#             num_attr > 0,
#             attr_mass / num_attr,
#             torch.zeros_like(num_attr)
#         )

#         obj_prob = torch.where(
#             num_obj > 0,
#             obj_mass / num_obj,
#             torch.zeros_like(num_obj)
#         )

#         smooth_labels = (
#             same_attr.float() * attr_prob
#             + same_obj.float() * obj_prob
#         )

#         # --------------------------------------------------
#         # 5. GT probability
#         # --------------------------------------------------
#         smooth_labels.scatter_(
#             1,
#             target.unsqueeze(1),
#             1.0 - self.ls
#         )

#         # --------------------------------------------------
#         # 6. Cross entropy
#         # --------------------------------------------------
#         log_prob = F.log_softmax(input, dim=-1)

#         return -(smooth_labels * log_prob).sum(dim=-1).mean()
    
    
# class AdaptivePartialLabelSmoothing(nn.Module):
#     def __init__(
#         self,
#         train_pairs,
#         ls=0.025,
#         obj_ratio=0.7,
#         attr_ratio=0.3,
#         temperature=0.1,
#         gamma=1.0,
#     ):
#         super().__init__()

#         self.register_buffer(
#             "comp_pairs",
#             torch.as_tensor(train_pairs, dtype=torch.long)
#         )

#         self.ls = float(ls)
#         self.obj_ratio = float(obj_ratio)
#         self.attr_ratio = float(attr_ratio)
#         self.temperature = float(temperature)
#         self.gamma = float(gamma)

#     def forward(self, logits, target, attr_sim=None, obj_sim=None):

#         B, K = logits.shape
#         pairs = self.comp_pairs.to(logits.device)

#         # Make sure target is integer
#         target = target.long()

#         gt_attr = pairs[target, 0].unsqueeze(1)
#         gt_obj = pairs[target, 1].unsqueeze(1)

#         all_attr = pairs[:, 0].unsqueeze(0)
#         all_obj = pairs[:, 1].unsqueeze(0)

#         same_attr = all_attr == gt_attr
#         same_obj = all_obj == gt_obj

#         # Exclude GT
#         gt_mask = torch.zeros_like(same_attr)
#         gt_mask.scatter_(1, target.unsqueeze(1), True)

#         same_attr = same_attr & ~gt_mask
#         same_obj = same_obj & ~gt_mask

#         # ------------------------------------------------
#         # Confidence / uncertainty
#         # ------------------------------------------------
#         prob = F.softmax(logits.detach(), dim=-1)

#         entropy = -(prob * torch.log(prob.clamp_min(1e-8))).sum(dim=-1)

#         entropy = entropy / math.log(K)

#         # IMPORTANT: keep same dtype as logits
#         eps = (
#             self.ls * entropy.pow(self.gamma)
#         ).to(dtype=logits.dtype)

#         # ------------------------------------------------
#         # Relation weights
#         # ------------------------------------------------
#         if attr_sim is None:
#             attr_weight = same_attr.to(dtype=logits.dtype)
#         else:
#             attr_weight = attr_sim[target].to(
#                 device=logits.device,
#                 dtype=logits.dtype
#             )
#             attr_weight = attr_weight * same_attr.to(logits.dtype)

#         if obj_sim is None:
#             obj_weight = same_obj.to(dtype=logits.dtype)
#         else:
#             obj_weight = obj_sim[target].to(
#                 device=logits.device,
#                 dtype=logits.dtype
#             )
#             obj_weight = obj_weight * same_obj.to(logits.dtype)

#         # Normalize
#         attr_weight = attr_weight / (
#             attr_weight.sum(dim=-1, keepdim=True).clamp_min(1e-8)
#         )

#         obj_weight = obj_weight / (
#             obj_weight.sum(dim=-1, keepdim=True).clamp_min(1e-8)
#         )

#         # ------------------------------------------------
#         # Target distribution
#         # ------------------------------------------------
#         labels = torch.zeros_like(logits)

#         gt_prob = (1.0 - eps).unsqueeze(1)

#         labels.scatter_(
#             1,
#             target.unsqueeze(1),
#             gt_prob
#         )

#         attr_mass = (
#             eps * self.attr_ratio
#         ).unsqueeze(1)

#         obj_mass = (
#             eps * self.obj_ratio
#         ).unsqueeze(1)

#         labels += attr_mass * attr_weight
#         labels += obj_mass * obj_weight

#         # ------------------------------------------------
#         # CE
#         # ------------------------------------------------
#         log_prob = F.log_softmax(logits, dim=-1)

#         loss = -(labels * log_prob).sum(dim=-1)

#         return loss.mean()



# class BalancedPartialLabelSmoothing(nn.Module):
#     def __init__(
#         self,
#         train_pairs,
#         ls=0.02,
#         balance=0.8,
#     ):
#         super().__init__()

#         self.register_buffer(
#             "comp_pairs",
#             torch.as_tensor(train_pairs, dtype=torch.long)
#         )

#         self.ls = float(ls)

#         # Controls how strongly the object/attribute ratio
#         # is pulled toward 50/50.
#         #
#         # balance = 0.0 -> fully uncertainty-adaptive
#         # balance = 1.0 -> strongly pulled toward 50/50
#         self.balance = float(balance)

#     def forward(
#         self,
#         logits,
#         target,
#         attr_logits=None,
#         obj_logits=None,
#         attr_sim=None,
#         obj_sim=None,
#     ):

#         B, K = logits.shape

#         target = target.long()
#         pairs = self.comp_pairs.to(logits.device)

#         # ============================================================
#         # 1. Find same-attribute / same-object candidates
#         # ============================================================

#         gt_attr = pairs[target, 0].unsqueeze(1)
#         gt_obj = pairs[target, 1].unsqueeze(1)

#         all_attr = pairs[:, 0].unsqueeze(0)
#         all_obj = pairs[:, 1].unsqueeze(0)

#         same_attr = all_attr == gt_attr
#         same_obj = all_obj == gt_obj

#         # Remove ground-truth composition
#         gt_mask = torch.zeros_like(same_attr)
#         gt_mask.scatter_(1, target.unsqueeze(1), True)

#         same_attr = same_attr & ~gt_mask
#         same_obj = same_obj & ~gt_mask

#         # ============================================================
#         # 2. Fixed total smoothing mass
#         # ============================================================

#         eps = torch.full(
#             (B,),
#             self.ls,
#             device=logits.device,
#             dtype=logits.dtype,
#         )

#         # ============================================================
#         # 3. Adaptive object / attribute allocation
#         # ============================================================

#         if attr_logits is not None and obj_logits is not None:

#             # -------------------------------
#             # Attribute uncertainty
#             # -------------------------------

#             attr_prob = F.softmax(
#                 attr_logits.detach(),
#                 dim=-1
#             )

#             attr_entropy = -(
#                 attr_prob *
#                 torch.log(attr_prob.clamp_min(1e-8))
#             ).sum(dim=-1)

#             attr_entropy = attr_entropy / math.log(
#                 attr_logits.size(-1)
#             )

#             # -------------------------------
#             # Object uncertainty
#             # -------------------------------

#             obj_prob = F.softmax(
#                 obj_logits.detach(),
#                 dim=-1
#             )

#             obj_entropy = -(
#                 obj_prob *
#                 torch.log(obj_prob.clamp_min(1e-8))
#             ).sum(dim=-1)

#             obj_entropy = obj_entropy / math.log(
#                 obj_logits.size(-1)
#             )

#             # More uncertain branch gets more smoothing
#             obj_ratio = (
#                 obj_entropy + 1e-6
#             ) / (
#                 obj_entropy +
#                 attr_entropy +
#                 2e-6
#             )

#             attr_ratio = 1.0 - obj_ratio

#             # -------------------------------
#             # Pull toward balanced allocation
#             # -------------------------------

#             b = self.balance

#             obj_ratio = (
#                 obj_ratio + 0.5 * b
#             ) / (1.0 + b)

#             attr_ratio = (
#                 attr_ratio + 0.5 * b
#             ) / (1.0 + b)

#         else:

#             # Default: exactly balanced
#             obj_ratio = torch.full(
#                 (B,),
#                 0.5,
#                 device=logits.device,
#                 dtype=logits.dtype,
#             )

#             attr_ratio = torch.full(
#                 (B,),
#                 0.5,
#                 device=logits.device,
#                 dtype=logits.dtype,
#             )

#         # ============================================================
#         # 4. Semantic weights
#         # ============================================================

#         if attr_sim is None:

#             attr_weight = same_attr.to(logits.dtype)

#         else:

#             attr_weight = attr_sim[target].to(
#                 device=logits.device,
#                 dtype=logits.dtype
#             )

#             attr_weight = (
#                 attr_weight *
#                 same_attr.to(logits.dtype)
#             )

#         if obj_sim is None:

#             obj_weight = same_obj.to(logits.dtype)

#         else:

#             obj_weight = obj_sim[target].to(
#                 device=logits.device,
#                 dtype=logits.dtype
#             )

#             obj_weight = (
#                 obj_weight *
#                 same_obj.to(logits.dtype)
#             )

#         # Normalize candidate weights
#         attr_weight = attr_weight / (
#             attr_weight.sum(
#                 dim=-1,
#                 keepdim=True
#             ).clamp_min(1e-8)
#         )

#         obj_weight = obj_weight / (
#             obj_weight.sum(
#                 dim=-1,
#                 keepdim=True
#             ).clamp_min(1e-8)
#         )

#         # ============================================================
#         # 5. Construct target distribution
#         # ============================================================

#         labels = torch.zeros_like(logits)

#         # Ground truth
#         labels.scatter_(
#             1,
#             target.unsqueeze(1),
#             (1.0 - eps).unsqueeze(1)
#         )

#         # Attribute alternatives
#         attr_mass = (
#             eps * attr_ratio
#         ).unsqueeze(1)

#         labels += (
#             attr_mass *
#             attr_weight
#         )

#         # Object alternatives
#         obj_mass = (
#             eps * obj_ratio
#         ).unsqueeze(1)

#         labels += (
#             obj_mass *
#             obj_weight
#         )

#         # ============================================================
#         # 6. Cross entropy
#         # ============================================================

#         log_prob = F.log_softmax(
#             logits,
#             dim=-1
#         )

#         loss = -(
#             labels * log_prob
#         ).sum(dim=-1)

#         return loss.mean()
    



class DropoutAwarePartialLabelSmoothing(nn.Module):
    """
    Dropout-aware partial label smoothing for CZSL.

    Total smoothing mass is fixed:
        epsilon = ls

    The smoothing mass is divided between:
        - same-attribute compositions
        - same-object compositions

    Allocation is based on:
        1. Attribute/object prediction entropy
        2. JS disagreement between two dropout views

    Within each group, smoothing can optionally be
    weighted by semantic similarity.
    """

    def __init__(
        self,
        train_pairs,
        ls=0.025,
        dropout_weight=0.5,
        balance=1.0,
    ):
        super().__init__()

        self.register_buffer(
            "comp_pairs",
            torch.as_tensor(
                train_pairs,
                dtype=torch.long
            )
        )

        self.ls = float(ls)

        # Weight of JS disagreement relative to entropy.
        self.dropout_weight = float(dropout_weight)

        # Pull adaptive ratio toward 0.5 / 0.5.
        #
        # 0.0 -> fully adaptive
        # 1.0 -> strong balancing
        self.balance = float(balance)

    @staticmethod
    def entropy_from_logits(logits):
        """
        Normalized entropy in [0, 1].
        """
        prob = F.softmax(
            logits.detach(),
            dim=-1
        )

        entropy = -(
            prob *
            torch.log(
                prob.clamp_min(1e-8)
            )
        ).sum(dim=-1)

        entropy = entropy / math.log(
            logits.size(-1)
        )

        return entropy

    @staticmethod
    def js_disagreement(logits1, logits2):
        """
        Jensen-Shannon divergence between two
        predictive distributions.

        Returns a value >= 0 for each sample.

        The result is normalized by log(2), so the
        maximum JS divergence is approximately 1.
        """

        p = F.softmax(
            logits1.detach(),
            dim=-1
        )

        q = F.softmax(
            logits2.detach(),
            dim=-1
        )

        m = 0.5 * (p + q)

        log_p = torch.log(
            p.clamp_min(1e-8)
        )

        log_q = torch.log(
            q.clamp_min(1e-8)
        )

        log_m = torch.log(
            m.clamp_min(1e-8)
        )

        kl_pm = (
            p *
            (log_p - log_m)
        ).sum(dim=-1)

        kl_qm = (
            q *
            (log_q - log_m)
        ).sum(dim=-1)

        js = 0.5 * (
            kl_pm + kl_qm
        )

        # Normalize JS to approximately [0, 1].
        js = js / math.log(2.0)

        return js

    def forward(
        self,
        logits,
        target,
        attr_logits_1=None,
        attr_logits_2=None,
        obj_logits_1=None,
        obj_logits_2=None,
        attr_sim=None,
        obj_sim=None,
    ):

        B, K = logits.shape

        target = target.long()

        pairs = self.comp_pairs.to(
            logits.device
        )

        # ============================================================
        # 1. Same-attribute / same-object candidate groups
        # ============================================================

        gt_attr = pairs[
            target, 0
        ].unsqueeze(1)

        gt_obj = pairs[
            target, 1
        ].unsqueeze(1)

        all_attr = pairs[
            :, 0
        ].unsqueeze(0)

        all_obj = pairs[
            :, 1
        ].unsqueeze(0)

        same_attr = (
            all_attr == gt_attr
        )

        same_obj = (
            all_obj == gt_obj
        )

        # Remove ground truth
        gt_mask = torch.zeros_like(
            same_attr
        )

        gt_mask.scatter_(
            1,
            target.unsqueeze(1),
            True
        )

        same_attr = (
            same_attr &
            ~gt_mask
        )

        same_obj = (
            same_obj &
            ~gt_mask
        )

        # ============================================================
        # 2. Fixed total smoothing mass
        # ============================================================

        eps = torch.full(
            (B,),
            self.ls,
            device=logits.device,
            dtype=logits.dtype,
        )

        # ============================================================
        # 3. Attribute/object uncertainty
        # ============================================================

        if (
            attr_logits_1 is not None
            and attr_logits_2 is not None
        ):

            # Average the two dropout views for entropy.
            attr_logits_mean = (
                attr_logits_1.detach() +
                attr_logits_2.detach()
            ) * 0.5

            attr_entropy = (
                self.entropy_from_logits(
                    attr_logits_mean
                )
            )

            attr_js = self.js_disagreement(
                attr_logits_1,
                attr_logits_2
            )

        elif attr_logits_1 is not None:

            attr_entropy = (
                self.entropy_from_logits(
                    attr_logits_1
                )
            )

            attr_js = torch.zeros_like(
                attr_entropy
            )

        else:

            attr_entropy = torch.full(
                (B,),
                0.5,
                device=logits.device,
                dtype=logits.dtype,
            )

            attr_js = torch.zeros_like(
                attr_entropy
            )

        if (
            obj_logits_1 is not None
            and obj_logits_2 is not None
        ):

            obj_logits_mean = (
                obj_logits_1.detach() +
                obj_logits_2.detach()
            ) * 0.5

            obj_entropy = (
                self.entropy_from_logits(
                    obj_logits_mean
                )
            )

            obj_js = self.js_disagreement(
                obj_logits_1,
                obj_logits_2
            )

        elif obj_logits_1 is not None:

            obj_entropy = (
                self.entropy_from_logits(
                    obj_logits_1
                )
            )

            obj_js = torch.zeros_like(
                obj_entropy
            )

        else:

            obj_entropy = torch.full(
                (B,),
                0.5,
                device=logits.device,
                dtype=logits.dtype,
            )

            obj_js = torch.zeros_like(
                obj_entropy
            )

        # ============================================================
        # 4. Combined uncertainty
        # ============================================================

        attr_score = (
            attr_entropy +
            self.dropout_weight * attr_js
        )

        obj_score = (
            obj_entropy +
            self.dropout_weight * obj_js
        )

        # ============================================================
        # 5. Adaptive attribute/object allocation
        # ============================================================

        total_score = (
            attr_score +
            obj_score +
            1e-8
        )

        attr_ratio = (
            attr_score /
            total_score
        )

        obj_ratio = (
            obj_score /
            total_score
        )

        # ============================================================
        # 6. Balance the two sources
        # ============================================================

        b = self.balance

        attr_ratio = (
            attr_ratio +
            0.5 * b
        ) / (1.0 + b)

        obj_ratio = (
            obj_ratio +
            0.5 * b
        ) / (1.0 + b)

        # Numerical normalization
        ratio_sum = (
            attr_ratio +
            obj_ratio
        ).clamp_min(1e-8)

        attr_ratio = (
            attr_ratio / ratio_sum
        )

        obj_ratio = (
            obj_ratio / ratio_sum
        )

        # ============================================================
        # 7. Semantic weights inside each group
        # ============================================================

        if attr_sim is None:

            attr_weight = same_attr.to(
                dtype=logits.dtype
            )

        else:

            attr_weight = attr_sim[
                target
            ].to(
                device=logits.device,
                dtype=logits.dtype
            )

            attr_weight = (
                attr_weight *
                same_attr.to(
                    logits.dtype
                )
            )

        if obj_sim is None:

            obj_weight = same_obj.to(
                dtype=logits.dtype
            )

        else:

            obj_weight = obj_sim[
                target
            ].to(
                device=logits.device,
                dtype=logits.dtype
            )

            obj_weight = (
                obj_weight *
                same_obj.to(
                    logits.dtype
                )
            )

        # Normalize
        attr_weight = (
            attr_weight /
            attr_weight.sum(
                dim=-1,
                keepdim=True
            ).clamp_min(1e-8)
        )

        obj_weight = (
            obj_weight /
            obj_weight.sum(
                dim=-1,
                keepdim=True
            ).clamp_min(1e-8)
        )

        # ============================================================
        # 8. Construct partial-label distribution
        # ============================================================

        labels = torch.zeros_like(
            logits
        )

        # Ground truth
        labels.scatter_(
            1,
            target.unsqueeze(1),
            (1.0 - eps).unsqueeze(1)
        )

        # Attribute alternatives
        attr_mass = (
            eps * attr_ratio
        ).unsqueeze(1)

        labels += (
            attr_mass *
            attr_weight
        )

        # Object alternatives
        obj_mass = (
            eps * obj_ratio
        ).unsqueeze(1)

        labels += (
            obj_mass *
            obj_weight
        )

        # ============================================================
        # 9. Cross entropy
        # ============================================================

        log_prob = F.log_softmax(
            logits,
            dim=-1
        )

        loss = -(
            labels * log_prob
        ).sum(dim=-1)

        return loss.mean()

