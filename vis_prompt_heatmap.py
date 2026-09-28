"""
Prompt-conditioned Grad-CAM visualization for CAMS.

Expected workflow:
1) Train CAMS and save val_best.pt + val_best_prompt_bank.pt using the patches in the ChatGPT answer.
2) Run this file from your project root, e.g.

python tools/vis_prompt_heatmap.py \
  --config configs/your_config.yml \
  --checkpoint path/to/save/val_best.pt \
  --sample-index 0 \
  --phase test \
  --branch comp \
  --out path/to/save/prompt_heatmap.png

This script assumes your dataset item returns at least:
    image_tensor, attr_id, obj_id, pair_id, ...
If your dataset uses a different normalization than CLIP, edit CLIP_MEAN and CLIP_STD.
"""

import argparse
import math
import os
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from flags import parser as project_parser
from utils import load_args, set_seed
from dataset import CompositionDataset
from model.configure_model import configure_model


CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def load_checkpoint_compat(model, checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    # Compatible with checkpoints saved from DataParallel or non-DataParallel.
    new_state = {}
    for k, v in state.items():
        if k.startswith("module."):
            new_state[k[len("module."):]] = v
        else:
            new_state[k] = v
    model.load_state_dict(new_state, strict=True)
    return ckpt


def choose_target_layer(core_model):
    """Auto-pick a Grad-CAM layer for common CLIP visual backbones."""
    visual = core_model.clip_model.visual
    if hasattr(visual, "layer4"):
        return visual.layer4[-1]
    if hasattr(visual, "transformer") and hasattr(visual.transformer, "resblocks"):
        # ViT: hook the last block norm/residual output. If your CLIP variant exposes
        # a better patch-token layer, replace this with that layer.
        return visual.transformer.resblocks[-1].ln_1
    raise ValueError(
        "Cannot auto-detect target_layer. Manually choose a layer inside "
        "model.clip_model.visual, e.g. visual.layer4[-1] for ResNet-CLIP or "
        "visual.transformer.resblocks[-1].ln_1 for ViT-CLIP."
    )


class PromptGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.fwd_handle = target_layer.register_forward_hook(self._forward_hook)
        self.bwd_handle = target_layer.register_full_backward_hook(self._backward_hook)

    def close(self):
        self.fwd_handle.remove()
        self.bwd_handle.remove()

    def _forward_hook(self, module, inputs, output):
        self.activations = output

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    @staticmethod
    def _normalize_feature(x):
        return F.normalize(x, p=2, dim=-1)

    def _score(self, img, prompt_bank: Dict, prompt_key: str, prompt_index: int, branch: str):
        core = unwrap_model(self.model)
        att, obj, com, glb = core.image_encoder(img)
        temp = core.temp_logit

        if branch == "attr":
            img_feat = self._normalize_feature(att)
        elif branch == "obj":
            img_feat = self._normalize_feature(obj)
        elif branch == "glb":
            img_feat = self._normalize_feature(glb)
        else:
            img_feat = self._normalize_feature(com)

        txt_feat = prompt_bank["text_features"][prompt_key].to(device=img.device, dtype=img_feat.dtype)
        txt_feat = txt_feat[prompt_index].unsqueeze(0)
        temp = temp.to(dtype=img_feat.dtype)
        return (temp * img_feat @ txt_feat.T).squeeze()

    def generate(self, img, prompt_bank: Dict, prompt_key: str, prompt_index: int, branch: str):
        self.model.zero_grad(set_to_none=True)
        score = self._score(img, prompt_bank, prompt_key, prompt_index, branch)
        score.backward(retain_graph=False)
        cam = self._make_cam(img)
        return cam, float(score.detach().cpu().item())

    def _make_cam(self, img):
        acts = self.activations
        grads = self.gradients
        if acts is None or grads is None:
            raise RuntimeError("Hooks did not capture activations/gradients. Check target_layer.")

        # CNN feature map: [B, C, H, W]
        if acts.dim() == 4:
            weights = grads.mean(dim=(2, 3), keepdim=True)
            cam = (weights * acts).sum(dim=1)
            cam = F.relu(cam)

        # ViT tokens: [B, N, C] or [N, B, C]
        elif acts.dim() == 3:
            if acts.shape[0] != img.shape[0] and acts.shape[1] == img.shape[0]:
                acts = acts.permute(1, 0, 2)
                grads = grads.permute(1, 0, 2)
            tokens = acts[:, 1:, :] if acts.shape[1] > 1 else acts
            token_grads = grads[:, 1:, :] if grads.shape[1] == acts.shape[1] and grads.shape[1] > 1 else grads
            n_tokens = tokens.shape[1]
            h = int(math.sqrt(n_tokens))
            if h * h != n_tokens:
                raise RuntimeError(
                    f"Cannot reshape {n_tokens} ViT tokens into a square map. "
                    "Use a different target_layer that preserves patch tokens."
                )
            weights = token_grads.mean(dim=1, keepdim=True)
            cam_tokens = (weights * tokens).sum(dim=-1)
            cam = F.relu(cam_tokens).view(img.shape[0], h, h)
        else:
            raise RuntimeError(f"Unsupported activation shape: {tuple(acts.shape)}")

        cam = F.interpolate(cam.unsqueeze(1), size=img.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze(1)
        cam_min = cam.flatten(1).min(dim=1)[0].view(-1, 1, 1)
        cam_max = cam.flatten(1).max(dim=1)[0].view(-1, 1, 1)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-6)
        return cam[0].detach().cpu().numpy()


def tensor_to_image(img_tensor):
    x = img_tensor.detach().cpu()[0]
    x = (x * CLIP_STD + CLIP_MEAN).clamp(0, 1)
    return x.permute(1, 2, 0).numpy()


def save_heatmap_panel(image_np, results: Dict[str, Tuple[np.ndarray, float]], out_path):
    n = len(results) + 1
    plt.figure(figsize=(4 * n, 4))
    plt.subplot(1, n, 1)
    plt.imshow(image_np)
    plt.title("image")
    plt.axis("off")

    for i, (name, (cam, score)) in enumerate(results.items(), start=2):
        plt.subplot(1, n, i)
        plt.imshow(image_np)
        plt.imshow(cam, alpha=0.45, cmap="jet")
        plt.title(f"{name}\nscore={score:.3f}")
        plt.axis("off")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def parse_args():
    vis_parser = argparse.ArgumentParser(add_help=False)
    vis_parser.add_argument("--checkpoint", required=True)
    vis_parser.add_argument("--sample-index", type=int, default=0)
    vis_parser.add_argument("--phase", default="test", choices=["train", "val", "test"])
    vis_parser.add_argument("--branch", default="comp", choices=["comp", "glb", "attr", "obj"])
    vis_parser.add_argument("--out", default="prompt_heatmap.png")
    vis_args, remaining = vis_parser.parse_known_args()

    args = project_parser.parse_args(remaining)
    if args.config:
        load_args(args.config, args)
    return vis_args, args


def main():
    vis_args, args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)

    train_dataset = CompositionDataset(
        args.dataset_path,
        phase="train",
        split="compositional-split-natural",
        open_world=args.open_world,
    )
    sample_dataset = CompositionDataset(
        args.dataset_path,
        phase=vis_args.phase,
        split="compositional-split-natural",
        open_world=args.open_world,
    )

    model, _ = configure_model(args, train_dataset)
    model.to(device)
    load_checkpoint_compat(model, vis_args.checkpoint, device)
    core = unwrap_model(model)
    core.eval()

    sample = sample_dataset[vis_args.sample_index]
    img, attr_id, obj_id = sample[0], int(sample[1]), int(sample[2])
    img = img.unsqueeze(0).to(device)

    # Build a one-pair prompt bank for the sample pair. This also works for unseen pairs.
    pair_idx = torch.tensor([[attr_id, obj_id]], dtype=torch.long, device=device)
    with torch.no_grad():
        prompt_bank = core.export_prompt_bank(pair_idx=pair_idx, include_token_embeddings=False)

    if vis_args.branch in ["comp", "glb"]:
        prompt_keys = ["comp_pos", "comp_neg_att", "comp_neg_obj", "comp_neg_both"]
        prompt_index = 0
        branch = vis_args.branch
    elif vis_args.branch == "attr":
        prompt_keys = ["attr_pos", "attr_neg"]
        prompt_index = attr_id
        branch = "attr"
    else:
        prompt_keys = ["obj_pos", "obj_neg"]
        prompt_index = obj_id
        branch = "obj"

    target_layer = choose_target_layer(core)
    cam = PromptGradCAM(model, target_layer)
    try:
        results = {}
        for key in prompt_keys:
            heat, score = cam.generate(img, prompt_bank, key, prompt_index, branch)
            results[key] = (heat, score)
    finally:
        cam.close()

    image_np = tensor_to_image(img)
    save_heatmap_panel(image_np, results, vis_args.out)
    print(f"Saved prompt heatmap to: {vis_args.out}")


if __name__ == "__main__":
    main()
