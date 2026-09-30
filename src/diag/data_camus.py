"""CAMUS processed data access (read-only inputs, local outputs).

Deviation record (also written to mlflow tags by callers):
- dts: processed PNGs carry no timestamp metadata -> ones (no real intervals).
  dt path proven live in smoke; =1 reduces Euler to residual update.
- ED/ES: placeholder = frame index order; paper's ED/ES definition pending.
  Callers needing ED/ES-Mean must treat values as approximate.
- Labels: gt_lv PNGs {0,1}; some frames may be empty (no cavity visible).
"""
from __future__ import annotations

from pathlib import Path

import os

import numpy as np
import torch
from PIL import Image

PROCESSED = Path(os.environ.get("DIAG_CAMUS", "/input0/processed/camus_png256_10f"))


def load_patient(
    pid: str,
    root: Path = PROCESSED,
    frames: int = 10,
    size: int = 256,
) -> dict[str, torch.Tensor]:
    """Load one patient clip: images (T,1,1,H,W) float[0,1], masks (T,1,1,H,W) 0/1."""
    imgs = sorted((root / "img" / pid).glob("*.png"))[:frames]
    gts = sorted((root / "gt_lv" / pid).glob("*.png"))[:frames]
    assert len(imgs) == frames and len(gts) == frames, (pid, len(imgs), len(gts))
    x = torch.stack(
        [torch.from_numpy(np.array(Image.open(f))).float().div(255) for f in imgs]
    ).unsqueeze(1).unsqueeze(1)  # (T,1,1,H,W)
    g = torch.stack(
        [torch.from_numpy(np.array(Image.open(f))).float() for f in gts]
    ).unsqueeze(1).unsqueeze(1)
    return {"images": x, "masks": (g > 0.5).float(), "pid": pid,
            "dts": torch.ones(frames - 1)}


def load_split(split: str = "train_data", root: Path = PROCESSED) -> list[str]:
    import json

    meta = json.loads((root / "camus_public_datasplit_20250706.json").read_text())
    return meta[split]


DEVIATIONS = {
    "dts": "ones-no-timestamp-DEVIATION",
    "ed_es": "index-order-placeholder-DEVIATION",
    "lambda": "placeholder-supplement-pending-DEVIATION",
    "backbone": "UNeXtOfficialBackbone-random-init-DEVIATION",
}
