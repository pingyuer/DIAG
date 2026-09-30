"""本地 CPU smoke：读一小 clip（10 帧 PNG）+ Dummy 指标 + mlflow run 打标 code_sha。"""
import json
import sys
import urllib.request
from pathlib import Path

import mlflow
import numpy as np
from PIL import Image

MLFLOW_URI = "http://172.16.240.77:5000"
CLIP_DIR = Path("outputs/smoke_sample")  # pull 或 scp 来的单 clip 样本（10 帧）
REMOTE_CLIP = "/input0/processed/camus_png256_10f/img/patient0001"
SSH = "ssh -o BatchMode=yes -p 32237 root@172.16.240.188"


def ensure_sample():
    if len(list(CLIP_DIR.glob("*.png"))) >= 10:
        return
    import subprocess
    CLIP_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        f"{SSH} 'ls {REMOTE_CLIP}'", shell=True, check=True, capture_output=True)
    subprocess.run(
        f"scp -o BatchMode=yes -P 32237 root@172.16.240.188:{REMOTE_CLIP}/*.png {CLIP_DIR}/",
        shell=True, check=True)
    print(f"sample fetched -> {CLIP_DIR}", flush=True)


def main():
    ensure_sample()
    frames = sorted(CLIP_DIR.glob("*.png"))[:10]
    assert len(frames) == 10, f"期望 10 帧，实际 {len(frames)}"
    arr = np.stack([np.asarray(Image.open(f)) for f in frames])  # (10,256,256)
    assert arr.shape == (10, 256, 256), arr.shape
    # Dummy 基线：阈值分割 + 帧间 Dice（动力学占位，后续 DIAG 替换）
    masks = (arr > arr.mean(axis=(1, 2), keepdims=True)).astype(np.uint8)
    inter = np.logical_and(masks[:-1], masks[1:]).sum(axis=(1, 2))
    denom = masks[:-1].sum(axis=(1, 2)) + masks[1:].sum(axis=(1, 2))
    dice = float((2 * inter / np.maximum(denom, 1)).mean())
    print(f"frames={arr.shape} dummy_interframe_dice={dice:.4f}", flush=True)

    code_sha = Path(".code_sha").read_text().strip()
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("diag-env-bringup")
    with mlflow.start_run(run_name="smoke-local-cpu"):
        mlflow.set_tag("code_sha", code_sha)
        mlflow.set_tag("host", "local-cpu")
        mlflow.log_param("frames", 10)
        mlflow.log_param("frame_size", 256)
        mlflow.log_metric("dummy_interframe_dice", dice)
    print(f"mlflow run ok code_sha={code_sha}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
