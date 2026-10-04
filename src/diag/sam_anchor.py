"""Frozen SAM-family anchor: MedSAM/SAM vision_encoder -> (F_tf, F_tc).

Interface discovery (008-3):
- SamModel.vision_encoder(pixel_values=1024px) -> (B,256,64,64) dense map.
- 64x64 grid == F_tc scale exactly; F_tf = bilinear x2 to 128x128.
- 256px echo upsampled x4 to 1024 (recorded deviation: blur + 16x pixels,
  frozen encoder cost ~4x). 1x1 adapt 256->96 trainable, rest frozen.
- MedSAM vs SAM verdict bar: C beats D else medical-pretraining unproven.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

SAM_MODELS = {
    "medsam": "wanglab/medsam-vit-base",
    "sam": "facebook/sam-vit-base",
}


class FrozenSamAnchor(nn.Module):
    """Pretrained frozen SAM-family encoder -> (F_tf, F_tc), C-channel."""

    def __init__(self, name: str = "medsam", out_channels: int = 96) -> None:
        super().__init__()
        from transformers import AutoModel

        model_id = SAM_MODELS[name]
        self.name = name
        self.encoder = AutoModel.from_pretrained(model_id, local_files_only=True)
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.adapt = nn.Conv2d(256, out_channels, 1)
        self.down = nn.AvgPool2d(2, 2)
        self.out_channels = out_channels

    def encode_frames(self, image: torch.Tensor) -> torch.Tensor:
        """(B,1,256,256) gray -> adapted map (B,C,128,128). Frozen backbone."""
        x = image.expand(-1, 3, -1, -1)
        x1024 = F.interpolate(x, size=(1024, 1024), mode="bilinear",
                              align_corners=False)
        with torch.no_grad():
            h = self.encoder.vision_encoder(pixel_values=x1024).last_hidden_state
        up = F.interpolate(h, size=(128, 128), mode="bilinear", align_corners=False)
        return self.adapt(up)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        f_tf = self.encode_frames(image)
        return {"F_tf": f_tf, "F_tc": self.down(f_tf)}

    def content(self, image: torch.Tensor) -> torch.Tensor:
        return self.forward(image)["F_tf"]

    def observe(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.forward(image)
        return out["F_tf"], out["F_tc"]
