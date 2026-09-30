"""Anchoring smoke：真 clip 双尺度 Eq.5 + mlflow run 打标 code_sha。"""
import subprocess
import sys
import time
from pathlib import Path

import mlflow
import numpy as np
import torch
from PIL import Image

from diag.anchoring import BACKBONE_RECORD, ContentAnchor

MLFLOW_URI = "http://172.16.240.77:5000"


def main():
    t0 = time.time()
    torch.manual_seed(0)
    model = ContentAnchor().eval()
    frames = sorted(Path("outputs/smoke_sample").glob("*.png"))[:10]
    assert len(frames) == 10
    clip = torch.stack(
        [torch.from_numpy(np.asarray(Image.open(f))).float().div(255) for f in frames]
    ).unsqueeze(1)
    with torch.no_grad():
        outs = [model(frame.unsqueeze(0)) for frame in clip]
    f_tf = torch.stack([o["F_tf"] for o in outs])
    f_tc = torch.stack([o["F_tc"] for o in outs])
    assert f_tf.shape == (10, 1, 96, 128, 128), f_tf.shape
    assert f_tc.shape == (10, 1, 96, 64, 64), f_tc.shape
    down = torch.nn.functional.avg_pool2d(f_tf.flatten(0, 1), 2, 2).view_as(f_tc)
    maxdiff = float((down - f_tc).abs().max())
    assert maxdiff == 0.0, maxdiff
    n_params = sum(p.numel() for p in model.parameters())
    dt = time.time() - t0
    print(f"F_tf={tuple(f_tf.shape)} F_tc={tuple(f_tc.shape)} Down==F_tc maxdiff={maxdiff}", flush=True)
    print(f"params={n_params} time={dt:.2f}s backbone={BACKBONE_RECORD['name']}", flush=True)

    code_sha = subprocess.run(
        ["./DIAG-code/sync.sh", "sha"], capture_output=True, text=True, check=True
    )
    sha = Path(".code_sha").read_text().strip()
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("diag-env-bringup")
    with mlflow.start_run(run_name="smoke-anchor-eq5"):
        mlflow.set_tag("code_sha", sha)
        mlflow.set_tag("host", "local-cpu")
        mlflow.set_tag("node", "anchoring-eq5")
        mlflow.log_param("backbone", BACKBONE_RECORD["name"])
        mlflow.log_param("pretrained", BACKBONE_RECORD["pretrained"])
        mlflow.log_metric("down_consistency_maxdiff", maxdiff)
        mlflow.log_metric("n_params", float(n_params))
    print(f"mlflow run ok code_sha={sha}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
