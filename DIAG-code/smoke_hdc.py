"""HDC smoke: Anchor->PCLF->HDC end-to-end on real 10-frame clip + mlflow."""
import subprocess
import sys
from pathlib import Path

import mlflow
import numpy as np
import torch
from PIL import Image

from diag.anchoring import ContentAnchor
from diag.hdc import HDC, adjacent_ratio_cv
from diag.pclf import PCLF

MLFLOW_URI = "http://172.16.240.77:5000"


def main():
    torch.manual_seed(0)
    anchor = ContentAnchor().eval()
    pclf = PCLF(anchor.out_channels).eval()
    hdc = HDC(anchor.out_channels, num_queries=4).eval()
    frames = sorted(Path("outputs/smoke_sample").glob("*.png"))[:10]
    assert len(frames) == 10
    clip = torch.stack(
        [torch.from_numpy(np.asarray(Image.open(f))).float().div(255) for f in frames]
    ).unsqueeze(1)
    with torch.no_grad():
        feats = [anchor(frame.unsqueeze(0)) for frame in clip]
    f_tf = torch.stack([o["F_tf"] for o in feats])
    f_tc = torch.stack([o["F_tc"] for o in feats])
    with torch.no_grad():
        s_tf, s_tc = pclf.states(f_tf, f_tc)
        out = hdc(f_tf, s_tf, s_tc, record_gamma=True)
    assert out["F_e"].shape == f_tf.shape, (out["F_e"].shape, f_tf.shape)
    assert out["P"].shape == f_tf.shape
    assert out["Q_e"].shape == (10, 1, 4, 96), out["Q_e"].shape
    g = out["gamma"]
    assert bool((g > 0).all()), "gamma must stay positive"
    ac, af = float(out["alpha_c"]), float(out["alpha_f"])
    assert 0 < ac < 1 and 0 < af < 1, (ac, af)
    # gamma archive -> CV diagnostic path (untrained: smoothness only, no target value)
    cv = adjacent_ratio_cv([gg.detach() for gg in out["gamma_log"][0].unbind(0)])
    n_params = sum(p.numel() for p in hdc.parameters())
    print(f"F_e={tuple(out['F_e'].shape)} P={tuple(out['P'].shape)} Q_e={tuple(out['Q_e'].shape)}", flush=True)
    print(f"gamma in ({float(g.min()):.4f},{float(g.max()):.4f}) alpha=({ac:.3f},{af:.3f}) cv={cv['cv']:.2e} params={n_params}", flush=True)

    subprocess.run(["./DIAG-code/sync.sh", "sha"], capture_output=True, check=True)
    sha = Path(".code_sha").read_text().strip()
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("diag-env-bringup")
    with mlflow.start_run(run_name="smoke-hdc-eq1015"):
        mlflow.set_tag("code_sha", sha)
        mlflow.set_tag("host", "local-cpu")
        mlflow.set_tag("node", "hdc-eq1015")
        mlflow.log_param("num_queries", 4)
        mlflow.log_metric("gamma_min", float(g.min()))
        mlflow.log_metric("alpha_c", ac)
        mlflow.log_metric("n_params", float(n_params))
    print(f"mlflow run ok code_sha={sha}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
