"""PCLF smoke: Anchor->PCLF end-to-end on real 10-frame clip + dt checks + mlflow."""
import subprocess
import sys
from pathlib import Path

import mlflow
import numpy as np
import torch
from PIL import Image

from diag.anchoring import ContentAnchor
from diag.pclf import PCLF

MLFLOW_URI = "http://172.16.240.77:5000"


def main():
    torch.manual_seed(0)
    anchor = ContentAnchor().eval()
    pclf = PCLF(anchor.out_channels).eval()
    frames = sorted(Path("outputs/smoke_sample").glob("*.png"))[:10]
    assert len(frames) == 10
    clip = torch.stack(
        [torch.from_numpy(np.asarray(Image.open(f))).float().div(255) for f in frames]
    ).unsqueeze(1)  # (10,1,1,256,256)
    with torch.no_grad():
        feats = [anchor(frame.unsqueeze(0)) for frame in clip]
    f_tf = torch.stack([o["F_tf"] for o in feats])  # (10,1,96,128,128)
    f_tc = torch.stack([o["F_tc"] for o in feats])  # (10,1,96,64,64)
    with torch.no_grad():
        out = pclf(f_tf, f_tc)
    stf, stc = out["fine"]["states"], out["coarse"]["states"]
    assert stf.shape == f_tf.shape and stc.shape == f_tc.shape
    s1_ok = bool((stf[0] - out["fine"]["obs"][0]).abs().max() == 0)
    assert s1_ok
    # dt sensitivity with live vf weights
    with torch.no_grad():
        for flow in (pclf.flow_f, pclf.flow_c):
            flow.vf.net[-1].weight.normal_(std=0.02)
        a = pclf(f_tf, f_tc, dts=1.0)["fine"]["states"]
        b = pclf(f_tf, f_tc, dts=2.0)["fine"]["states"]
    dt_diff = float((a[1:] - b[1:]).abs().max())
    assert dt_diff > 0, "dt path dead"
    # causality
    with torch.no_grad():
        pert = f_tf.clone()
        pert[5] += 10.0
        base = pclf(f_tf, f_tc)["fine"]["states"]
        ps = pclf(pert, f_tc)["fine"]["states"]
    causal_ok = bool((base[:5] - ps[:5]).abs().max() == 0)
    assert causal_ok
    n_params = sum(p.numel() for p in pclf.parameters())
    print(f"S_tf={tuple(stf.shape)} S_tc={tuple(stc.shape)} S1==O1 gate_ok dt_diff={dt_diff:.4f}", flush=True)
    print(f"causal={causal_ok} params={n_params}", flush=True)

    subprocess.run(["./DIAG-code/sync.sh", "sha"], capture_output=True, check=True)
    sha = Path(".code_sha").read_text().strip()
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("diag-env-bringup")
    with mlflow.start_run(run_name="smoke-pclf-eq69"):
        mlflow.set_tag("code_sha", sha)
        mlflow.set_tag("host", "local-cpu")
        mlflow.set_tag("node", "pclf-eq69")
        mlflow.log_metric("dt_sensitivity_maxdiff", dt_diff)
        mlflow.log_metric("n_params", float(n_params))
    print(f"mlflow run ok code_sha={sha}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
