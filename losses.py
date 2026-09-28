import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ConfidenceWeightedPartialLabelSmoothing(nn.Module):
    """
    Partial-label smoothing weighted by CLIP zero-shot confidence.

    For a GT composition (a*, o*):
      * same-attribute candidates keep a*, but have alternative objects.
        Their weights are determined by the CLIP confidence of each
        candidate object relative to the GT object confidence.
      * same-object candidates keep o*, but have alternative attributes.
        Their weights are determined by the CLIP confidence of each
        candidate attribute relative to the GT attribute confidence.

    This avoids assigning the same smoothing probability to all related
    compositions. The GT itself always receives 1-ls.
    """

    def __init__(
        self,
        train_pairs,
        ls=0.025,
        balance=0.5,
        confidence_temperature=0.25,
        min_relation_weight=1e-4,
    ):
        super().__init__()
        self.register_buffer(
            "comp_pairs",
            torch.as_tensor(train_pairs, dtype=torch.long),
        )
        self.ls = float(ls)
        self.balance = float(balance)
        self.confidence_temperature = float(confidence_temperature)
        self.min_relation_weight = float(min_relation_weight)

    def _confidence_weights(
        self,
        confidence,
        gt_index,
        candidate_index,
        mask,
        dtype,
    ):
        """
        Weight candidates by their confidence relative to the GT.

        We use a sigmoid over the log-confidence gap rather than raw
        confidence. Therefore a candidate with confidence close to the
        GT receives substantial smoothing, while a much less plausible
        candidate receives very little.
        """
        confidence = confidence.clamp_min(1e-8)
        log_conf = torch.log(confidence)

        candidate_log_conf = log_conf.gather(
            1, candidate_index
        )
        gt_log_conf = log_conf.gather(
            1, gt_index
        )

        gap = (
            candidate_log_conf - gt_log_conf
        ) / max(self.confidence_temperature, 1e-6)

        weight = torch.sigmoid(gap)
        weight = weight.clamp_min(
            self.min_relation_weight
        )
        weight = weight * mask.to(dtype)

        return weight

    def forward(
        self,
        logits,
        target,
        attr_logits=None,
        obj_logits=None,
    ):
        B, K = logits.shape
        target = target.long()
        pairs = self.comp_pairs.to(logits.device)

        gt_attr = pairs[target, 0].unsqueeze(1)
        gt_obj = pairs[target, 1].unsqueeze(1)

        all_attr = pairs[:, 0].unsqueeze(0)
        all_obj = pairs[:, 1].unsqueeze(0)

        same_attr = all_attr == gt_attr
        same_obj = all_obj == gt_obj

        gt_mask = torch.zeros_like(same_attr)
        gt_mask.scatter_(1, target.unsqueeze(1), True)
        same_attr = same_attr & ~gt_mask
        same_obj = same_obj & ~gt_mask

        # ------------------------------------------------------------
        # Allocate the fixed smoothing mass between relation types.
        # ------------------------------------------------------------
        if attr_logits is not None and obj_logits is not None:
            attr_prob = F.softmax(attr_logits.detach(), dim=-1)
            obj_prob = F.softmax(obj_logits.detach(), dim=-1)

            gt_attr_conf = attr_prob.gather(
                1, gt_attr
            )
            gt_obj_conf = obj_prob.gather(
                1, gt_obj
            )

            # If the attribute is uncertain relative to the object,
            # allocate more mass to same-attribute alternatives.
            attr_uncertainty = 1.0 - gt_attr_conf.squeeze(1)
            obj_uncertainty = 1.0 - gt_obj_conf.squeeze(1)

            attr_ratio = (
                attr_uncertainty + 1e-6
            ) / (
                attr_uncertainty
                + obj_uncertainty
                + 2e-6
            )
            obj_ratio = 1.0 - attr_ratio

            # Pull toward a balanced relation allocation.
            b = self.balance
            attr_ratio = (
                attr_ratio + 0.5 * b
            ) / (1.0 + b)
            obj_ratio = (
                obj_ratio + 0.5 * b
            ) / (1.0 + b)
        else:
            attr_ratio = torch.full(
                (B,), 0.5, device=logits.device, dtype=logits.dtype
            )
            obj_ratio = torch.full(
                (B,), 0.5, device=logits.device, dtype=logits.dtype
            )

        # ------------------------------------------------------------
        # Confidence-weighted alternatives.
        #
        # Same attribute: the attribute is identical, so attribute
        # confidence cannot distinguish candidates. We therefore use
        # the candidate OBJECT confidence relative to the GT OBJECT.
        #
        # Same object: analogously use candidate ATTRIBUTE confidence
        # relative to the GT ATTRIBUTE.
        # ------------------------------------------------------------
        if attr_logits is not None and obj_logits is not None:
            attr_prob = F.softmax(attr_logits.detach(), dim=-1)
            obj_prob = F.softmax(obj_logits.detach(), dim=-1)

            candidate_obj = pairs[:, 1].unsqueeze(0).expand(B, -1)
            candidate_attr = pairs[:, 0].unsqueeze(0).expand(B, -1)

            gt_obj_idx = gt_obj
            gt_attr_idx = gt_attr

            same_attr_weight = self._confidence_weights(
                obj_prob,
                gt_obj_idx,
                candidate_obj,
                same_attr,
                logits.dtype,
            )

            same_obj_weight = self._confidence_weights(
                attr_prob,
                gt_attr_idx,
                candidate_attr,
                same_obj,
                logits.dtype,
            )
        else:
            same_attr_weight = same_attr.to(logits.dtype)
            same_obj_weight = same_obj.to(logits.dtype)

        # Normalize inside each relation group.
        same_attr_weight = same_attr_weight / (
            same_attr_weight.sum(dim=-1, keepdim=True)
            .clamp_min(1e-8)
        )
        same_obj_weight = same_obj_weight / (
            same_obj_weight.sum(dim=-1, keepdim=True)
            .clamp_min(1e-8)
        )

        # ------------------------------------------------------------
        # Target distribution
        # ------------------------------------------------------------
        labels = torch.zeros_like(logits)
        eps = torch.full(
            (B,), self.ls,
            device=logits.device,
            dtype=logits.dtype,
        )

        labels.scatter_(
            1,
            target.unsqueeze(1),
            (1.0 - eps).unsqueeze(1),
        )

        labels += (
            (eps * attr_ratio).unsqueeze(1)
            * same_attr_weight
        )
        labels += (
            (eps * obj_ratio).unsqueeze(1)
            * same_obj_weight
        )

        log_prob = F.log_softmax(logits, dim=-1)
        return -(labels * log_prob).sum(dim=-1).mean()
    
    
    
class PartialLabelSmoothing(nn.CrossEntropyLoss):
    def __init__(
        self,
        ls: float = 0,
        train_pairs=None,
        obj_ratio: float = 0.7,
        attr_ratio: float = 0.3,
        **kwargs
    ):
        super().__init__(**kwargs)

        assert 0 <= ls < 1
        assert abs(obj_ratio + attr_ratio - 1.0) < 1e-6

        self.ls = ls
        self.comp_pairs = train_pairs

        self.obj_ratio = obj_ratio
        self.attr_ratio = attr_ratio

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        """
        input:  (B, K) logits
        target: (B,) labels

        Label distribution:
            GT          : 1 - ls
            Same object : ls * obj_ratio
            Same attr   : ls * attr_ratio

        The smoothing mass is uniformly distributed within
        each relation group.
        """

        batch_size, total_class = input.size()

        # --------------------------------------------------
        # 1. Relation masks
        # --------------------------------------------------
        target_attr = self.comp_pairs[target, 0].unsqueeze(1)
        target_obj = self.comp_pairs[target, 1].unsqueeze(1)

        all_attr = self.comp_pairs[:, 0].unsqueeze(0)
        all_obj = self.comp_pairs[:, 1].unsqueeze(0)

        same_attr = (all_attr == target_attr)
        same_obj = (all_obj == target_obj)

        # Exclude GT itself from smoothing
        gt_mask = torch.zeros_like(same_attr)
        gt_mask.scatter_(1, target.unsqueeze(1), True)

        same_attr = same_attr & (~gt_mask)
        same_obj = same_obj & (~gt_mask)

        # --------------------------------------------------
        # 2. Count candidates
        # --------------------------------------------------
        num_attr = same_attr.sum(dim=1, keepdim=True).float()
        num_obj = same_obj.sum(dim=1, keepdim=True).float()

        # --------------------------------------------------
        # 3. Allocate smoothing mass
        # --------------------------------------------------
        attr_mass = self.ls * self.attr_ratio
        obj_mass = self.ls * self.obj_ratio

        # --------------------------------------------------
        # 4. Uniformly distribute within each group
        # --------------------------------------------------
        attr_prob = torch.where(
            num_attr > 0,
            attr_mass / num_attr,
            torch.zeros_like(num_attr)
        )

        obj_prob = torch.where(
            num_obj > 0,
            obj_mass / num_obj,
            torch.zeros_like(num_obj)
        )

        smooth_labels = (
            same_attr.float() * attr_prob
            + same_obj.float() * obj_prob
        )

        # --------------------------------------------------
        # 5. GT probability
        # --------------------------------------------------
        smooth_labels.scatter_(
            1,
            target.unsqueeze(1),
            1.0 - self.ls
        )

        # --------------------------------------------------
        # 6. Cross entropy
        # --------------------------------------------------
        log_prob = F.log_softmax(input, dim=-1)

        return -(smooth_labels * log_prob).sum(dim=-1).mean()
    
    
class AdaptivePartialLabelSmoothing(nn.Module):
    def __init__(
        self,
        train_pairs,
        ls=0.025,
        obj_ratio=0.7,
        attr_ratio=0.3,
        temperature=0.1,
        gamma=1.0,
    ):
        super().__init__()

        self.register_buffer(
            "comp_pairs",
            torch.as_tensor(train_pairs, dtype=torch.long)
        )

        self.ls = float(ls)
        self.obj_ratio = float(obj_ratio)
        self.attr_ratio = float(attr_ratio)
        self.temperature = float(temperature)
        self.gamma = float(gamma)

    def forward(self, logits, target, attr_sim=None, obj_sim=None):

        B, K = logits.shape
        pairs = self.comp_pairs.to(logits.device)

        # Make sure target is integer
        target = target.long()

        gt_attr = pairs[target, 0].unsqueeze(1)
        gt_obj = pairs[target, 1].unsqueeze(1)

        all_attr = pairs[:, 0].unsqueeze(0)
        all_obj = pairs[:, 1].unsqueeze(0)

        same_attr = all_attr == gt_attr
        same_obj = all_obj == gt_obj

        # Exclude GT
        gt_mask = torch.zeros_like(same_attr)
        gt_mask.scatter_(1, target.unsqueeze(1), True)

        same_attr = same_attr & ~gt_mask
        same_obj = same_obj & ~gt_mask

        # ------------------------------------------------
        # Confidence / uncertainty
        # ------------------------------------------------
        prob = F.softmax(logits.detach(), dim=-1)

        entropy = -(prob * torch.log(prob.clamp_min(1e-8))).sum(dim=-1)

        entropy = entropy / math.log(K)

        # IMPORTANT: keep same dtype as logits
        eps = (
            self.ls * entropy.pow(self.gamma)
        ).to(dtype=logits.dtype)

        # ------------------------------------------------
        # Relation weights
        # ------------------------------------------------
        if attr_sim is None:
            attr_weight = same_attr.to(dtype=logits.dtype)
        else:
            attr_weight = attr_sim[target].to(
                device=logits.device,
                dtype=logits.dtype
            )
            attr_weight = attr_weight * same_attr.to(logits.dtype)

        if obj_sim is None:
            obj_weight = same_obj.to(dtype=logits.dtype)
        else:
            obj_weight = obj_sim[target].to(
                device=logits.device,
                dtype=logits.dtype
            )
            obj_weight = obj_weight * same_obj.to(logits.dtype)

        # Normalize
        attr_weight = attr_weight / (
            attr_weight.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        )

        obj_weight = obj_weight / (
            obj_weight.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        )

        # ------------------------------------------------
        # Target distribution
        # ------------------------------------------------
        labels = torch.zeros_like(logits)

        gt_prob = (1.0 - eps).unsqueeze(1)

        labels.scatter_(
            1,
            target.unsqueeze(1),
            gt_prob
        )

        attr_mass = (
            eps * self.attr_ratio
        ).unsqueeze(1)

        obj_mass = (
            eps * self.obj_ratio
        ).unsqueeze(1)

        labels += attr_mass * attr_weight
        labels += obj_mass * obj_weight

        # ------------------------------------------------
        # CE
        # ------------------------------------------------
        log_prob = F.log_softmax(logits, dim=-1)

        loss = -(labels * log_prob).sum(dim=-1)

        return loss.mean()



class BalancedPartialLabelSmoothing(nn.Module):
    def __init__(
        self,
        train_pairs,
        ls=0.02,
        balance=0.8,
    ):
        super().__init__()

        self.register_buffer(
            "comp_pairs",
            torch.as_tensor(train_pairs, dtype=torch.long)
        )

        self.ls = float(ls)

        # Controls how strongly the object/attribute ratio
        # is pulled toward 50/50.
        #
        # balance = 0.0 -> fully uncertainty-adaptive
        # balance = 1.0 -> strongly pulled toward 50/50
        self.balance = float(balance)

    def forward(
        self,
        logits,
        target,
        attr_logits=None,
        obj_logits=None,
        attr_sim=None,
        obj_sim=None,
    ):

        B, K = logits.shape

        target = target.long()
        pairs = self.comp_pairs.to(logits.device)

        # ============================================================
        # 1. Find same-attribute / same-object candidates
        # ============================================================

        gt_attr = pairs[target, 0].unsqueeze(1)
        gt_obj = pairs[target, 1].unsqueeze(1)

        all_attr = pairs[:, 0].unsqueeze(0)
        all_obj = pairs[:, 1].unsqueeze(0)

        same_attr = all_attr == gt_attr
        same_obj = all_obj == gt_obj

        # Remove ground-truth composition
        gt_mask = torch.zeros_like(same_attr)
        gt_mask.scatter_(1, target.unsqueeze(1), True)

        same_attr = same_attr & ~gt_mask
        same_obj = same_obj & ~gt_mask

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
        # 3. Adaptive object / attribute allocation
        # ============================================================

        if attr_logits is not None and obj_logits is not None:

            # -------------------------------
            # Attribute uncertainty
            # -------------------------------

            attr_prob = F.softmax(
                attr_logits.detach(),
                dim=-1
            )

            attr_entropy = -(
                attr_prob *
                torch.log(attr_prob.clamp_min(1e-8))
            ).sum(dim=-1)

            attr_entropy = attr_entropy / math.log(
                attr_logits.size(-1)
            )

            # -------------------------------
            # Object uncertainty
            # -------------------------------

            obj_prob = F.softmax(
                obj_logits.detach(),
                dim=-1
            )

            obj_entropy = -(
                obj_prob *
                torch.log(obj_prob.clamp_min(1e-8))
            ).sum(dim=-1)

            obj_entropy = obj_entropy / math.log(
                obj_logits.size(-1)
            )

            # More uncertain branch gets more smoothing
            obj_ratio = (
                obj_entropy + 1e-6
            ) / (
                obj_entropy +
                attr_entropy +
                2e-6
            )

            attr_ratio = 1.0 - obj_ratio

            # -------------------------------
            # Pull toward balanced allocation
            # -------------------------------

            b = self.balance

            obj_ratio = (
                obj_ratio + 0.5 * b
            ) / (1.0 + b)

            attr_ratio = (
                attr_ratio + 0.5 * b
            ) / (1.0 + b)

        else:

            # Default: exactly balanced
            obj_ratio = torch.full(
                (B,),
                0.5,
                device=logits.device,
                dtype=logits.dtype,
            )

            attr_ratio = torch.full(
                (B,),
                0.5,
                device=logits.device,
                dtype=logits.dtype,
            )

        # ============================================================
        # 4. Semantic weights
        # ============================================================

        if attr_sim is None:

            attr_weight = same_attr.to(logits.dtype)

        else:

            attr_weight = attr_sim[target].to(
                device=logits.device,
                dtype=logits.dtype
            )

            attr_weight = (
                attr_weight *
                same_attr.to(logits.dtype)
            )

        if obj_sim is None:

            obj_weight = same_obj.to(logits.dtype)

        else:

            obj_weight = obj_sim[target].to(
                device=logits.device,
                dtype=logits.dtype
            )

            obj_weight = (
                obj_weight *
                same_obj.to(logits.dtype)
            )

        # Normalize candidate weights
        attr_weight = attr_weight / (
            attr_weight.sum(
                dim=-1,
                keepdim=True
            ).clamp_min(1e-8)
        )

        obj_weight = obj_weight / (
            obj_weight.sum(
                dim=-1,
                keepdim=True
            ).clamp_min(1e-8)
        )

        # ============================================================
        # 5. Construct target distribution
        # ============================================================

        labels = torch.zeros_like(logits)

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
        # 6. Cross entropy
        # ============================================================

        log_prob = F.log_softmax(
            logits,
            dim=-1
        )

        loss = -(
            labels * log_prob
        ).sum(dim=-1)

        return loss.mean()
    



