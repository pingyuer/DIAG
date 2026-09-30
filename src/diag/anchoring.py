"""DIAG Single-Frame Content Anchoring（论文 Eq.5）。

论文要求：目标帧 I_t 经**预训练图像编码器**抽取细尺度特征 F_tf 与
下采样粗尺度特征 F_tc = Down(F_tf)，x_t = (F_tf, F_tc)。

基座选型（偏差记录）：
- 论文要求预训练 segmenter（SAM/MedSAM/SAM2 系列，带 P0/Q0 接口）。
- 现实：无补充材料、无预训练权重可取；沿用 upstream UNeXt
  （`UNeXtOfficialBackbone`，与 DPFR 同款 backbone，随机初始化）。
- 偏差：UNeXt 随机初始化 ≠ 预训练内容锚，影响 H1（frame-only 缺口解释力）；
  待预训练权重到位后替换 backbone 即可，接口保持 (F_tf, F_tc) 不变。
- 权重来源：`src/diag/unext_backbone.py`（vendored 自
  `upstream_BanditPM/model/modules/unext/official.py`），torch 默认初始化，
  无外部 checkpoint。

接口：
- forward(I_t) -> {"F_tf", "F_tc"}：F_tf 取 backbone low（H/2），
  F_tc = Down(F_tf)（AvgPool2d×2，无参，与论文 Down 一致）。
- content(I_t) -> F_tf：内容路径（特征门控 + mask decoder 用）。
- observe(I_t) -> (F_tf, F_tc)：观测供给接口（后续 PCLF 观测编码器用）。
"""
from __future__ import annotations

import torch
import torch.nn as nn

from diag.unext_backbone import UNeXtOfficialBackbone

BACKBONE_RECORD = {
    "name": "UNeXtOfficialBackbone",
    "source": "vendored upstream_BanditPM/model/modules/unext/official.py",
    "pretrained": False,
    "checkpoint": None,
    "deviation": "论文要求预训练 segmenter；此处随机初始化，H1 相关结论记偏差",
}

CONTENT_KEY_FINE = "F_tf"
CONTENT_KEY_COARSE = "F_tc"


class ContentAnchor(nn.Module):
    """双尺度内容锚 Eq.5：x_t = (F_tf, F_tc=Down(F_tf))。"""

    def __init__(
        self,
        in_channels: int = 1,
        base_dim: int = 96,
        *,
        mlp_expansion: float = 2.0,
        latent_blocks: int = 2,
        decoder_mlp_blocks: int = 1,
    ) -> None:
        super().__init__()
        self.backbone = UNeXtOfficialBackbone(
            in_channels=in_channels,
            num_classes=2,
            base_dim=base_dim,
            value_dim=256,
            mlp_expansion=mlp_expansion,
            latent_blocks=latent_blocks,
            decoder_mlp_blocks=decoder_mlp_blocks,
        )
        self.down = nn.AvgPool2d(kernel_size=2, stride=2)
        self.out_channels = base_dim

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """image: (B, C, H, W) 单目标帧。返回 {F_tf: (B,C,H/2,W/2), F_tc: (B,C,H/4,W/4)}。"""
        if image.dim() != 4:
            raise ValueError(f"ContentAnchor expects BCHW, got {tuple(image.shape)}.")
        f_tf = self.backbone.encode(image)["low"]
        f_tc = self.down(f_tf)
        return {CONTENT_KEY_FINE: f_tf, CONTENT_KEY_COARSE: f_tc}

    def content(self, image: torch.Tensor) -> torch.Tensor:
        """内容路径：细尺度特征 F_tf。"""
        return self.forward(image)[CONTENT_KEY_FINE]

    def observe(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """观测供给接口：双尺度 (F_tf, F_tc)，供 PCLF 观测编码器 O^q 用。"""
        out = self.forward(image)
        return out[CONTENT_KEY_FINE], out[CONTENT_KEY_COARSE]
