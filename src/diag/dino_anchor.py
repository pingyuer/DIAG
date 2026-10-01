"""Frozen DINOv2 content anchor: pretrained dense features, (F_tf, F_tc) interface.

Why frozen first: isolates backbone QUALITY from dynamics learning. If frozen
DINO frame-only beats trained UNeXt frame-only, the gap is features, not
training. Unfreezing comes only after the frozen number is locked.

Design:
- DINOv2 patch14 on 256px -> 18x18 grid + CLS (325 tokens, dim 384/768).
  Drop CLS (global, no spatial meaning), reshape 18x18, bilinear to F_tf
  size (H/2=128). Bilinear on FEATURES (not masks) is safe: smooth semantic
  field, no ringing at this upsample ratio.
- 1x1 adapt dim -> C (default 96, matches ContentAnchor.out_channels so
  PCLF/HDC/Decoder need ZERO changes). Adapt is TRAINABLE (only it + decoder
  learn in frame-only runs; backbone frozen).
- Grayscale echo -> repeat to 3ch; DINO normalization (ImageNet mean/std).
  Recorded deviation: echo stats != ImageNet stats; revisit if zero-shot gaps.
- MedSAM/SAM (C/D candidates): same interface when weights arrive; P0/Q0
  extraction needs weight-code reading (not yet). This file takes backbone
  name; dinov2-small verified, -base downloaded, SAM family pending link.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

DINO_MEAN = (0.485, 0.456, 0.406)
DINO_STD = (0.229, 0.224, 0.225)

BACKBONES = {
    "dinov2-small": ("facebook/dinov2-small", 384),
    "dinov2-base": ("facebook/dinov2-base", 768),
}


class FrozenDinoAnchor(nn.Module):
    """Pretrained frozen encoder -> (F_tf, F_tc=Down(F_tf)), C-channel."""

    def __init__(self, name: str = "dinov2-small", out_channels: int = 96) -> None:
        super().__init__()
        from transformers import AutoModel

        model_id, dim = BACKBONES[name]
        self.name = name
        self.encoder = AutoModel.from_pretrained(model_id, local_files_only=True)
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.adapt = nn.Conv2d(dim, out_channels, 1)
        self.down = nn.AvgPool2d(2, 2)
        self.out_channels = out_channels
        mu = torch.tensor(DINO_MEAN).view(1, 3, 1, 1)
        sd = torch.tensor(DINO_STD).view(1, 3, 1, 1)
        self.register_buffer("mu", mu)
        self.register_buffer("sd", sd)

    def encode_frames(self, image: torch.Tensor) -> torch.Tensor:
        """(B,1,H,W) gray -> adapted dense map (B,C,H/2,W/2). Frozen backbone."""
        x = image.expand(-1, 3, -1, -1)
        x = (x - self.mu.to(x.device)) / self.sd.to(x.device)
        with torch.no_grad():
            h = self.encoder(pixel_values=x).last_hidden_state  # (B,325,D)
        patch = h[:, 1:, :].transpose(1, 2).reshape(x.shape[0], -1, 18, 18)
        up = F.interpolate(patch, size=(image.shape[-2] // 2, image.shape[-1] // 2),
                           mode="bilinear", align_corners=False)
        return self.adapt(up)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        f_tf = self.encode_frames(image)
        return {"F_tf": f_tf, "F_tc": self.down(f_tf)}

    def content(self, image: torch.Tensor) -> torch.Tensor:
        return self.forward(image)["F_tf"]

    def observe(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.forward(image)
        return out["F_tf"], out["F_tc"]
