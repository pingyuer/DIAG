"""Train/inference smoke: full Anchor->PCLF->HDC->Decoder->Loss on real clip.

Also asserts no-GT-leak: forward modules take only images (gt appears solely
as loss target); next-frame F_{t+1} used only inside L_flow.
"""
import inspect
import subprocess
import sys
from pathlib import Path

import mlflow
import numpy as np
import torch
from PIL import Image

from diag.anchoring import ContentAnchor
from diag.decoder import CandidateDecoder
from diag.hdc import HDC
from diag.losses import DiagLoss
from diag.pclf import PCLF

MLFLOW_URI = "http://172.16.240.77:5000"


def check_no_gt_leak() -> None:
    import diag.anchoring as a
    import diag.decoder as d
    import diag.hdc as h
    import diag.pclf as p
    src = "".join(
        inspect.getsource(m) for m in (a, p, h, d)
    ).lower()
    for tok in ("use_gt", "gt_prob", "teacher", "cls_gt"):
        assert tok not in src, f"GT-leak token in forward: {tok}"
    print("no-GT-leak: forward modules GT-free", flush=True)


def main():
    torch.manual_seed(0)
    check_no_gt_leak()
    anchor = ContentAnchor().eval()
    pclf = PCLF(anchor.out_channels)
    hdc = HDC(anchor.out_channels, num_queries=4)
    dec = CandidateDecoder(feat_dim=anchor.out_channels, num_queries=4, num_candidates=3)
    loss_fn = DiagLoss()
    frames = sorted(Path("outputs/smoke_sample").glob("*.png"))[:10]
    gt_frames = sorted(Path("outputs/smoke_sample").glob("*.png"))[:10]
    clip = torch.stack(
        [torch.from_numpy(np.asarray(Image.open(f))).float().div(255) for f in frames]
    ).unsqueeze(1)
    # pseudo-GT from threshold (smoke only, NOT fed to forward)
    with torch.no_grad():
        gray = clip.squeeze(2)
        gt_full = (gray > gray.mean(dim=(2, 3), keepdim=True)).float().unsqueeze(2)
    gt_small = torch.nn.functional.interpolate(
        gt_full.flatten(0, 1), size=(256, 256), mode="nearest").view(10, 1, 1, 256, 256)
    # decoder outputs at 256 (H/2 features -> 2x up); downsample gt to match
    with torch.no_grad():
        feats = [anchor(frame.unsqueeze(0)) for frame in clip]
    f_tf = torch.stack([o["F_tf"] for o in feats])
    f_tc = torch.stack([o["F_tc"] for o in feats])
    s_tf, s_tc = pclf.states(f_tf, f_tc)
    hout = hdc(f_tf, s_tf, s_tc)
    dout = dec(hout["F_e"], hout["P"], hout["Q_e"])
    mh, mw = dout["masks"].shape[-2:]
    gt = torch.nn.functional.interpolate(
        gt_full.flatten(0, 1), size=(mh, mw), mode="nearest").view(10, 1, 1, mh, mw)
    labeled = torch.ones(10, 1, dtype=torch.bool)
    labeled[5] = False  # one unlabeled frame: Omega masking works
    out = loss_fn(dout["masks"], dout["quality"], gt, labeled, s_tf,
                  pclf.forward(f_tf, f_tc)["fine"]["obs"])
    print(f"masks={tuple(dout['masks'].shape)} quality={tuple(dout['quality'].shape)}", flush=True)
    print("loss: " + " ".join(f"{k}={float(v.mean()):.4f}" for k, v in out.items() if k != "best"), flush=True)
    print(f"WTA best (first 5): {out['best'][:5].tolist()}", flush=True)
    # inference selection: argmax quality
    with torch.no_grad():
        sel = dout["quality"].argmax(dim=-1)  # (T,B)
    print(f"inference argmax quality (first 5): {sel[:5, 0].tolist()}", flush=True)
    out["total"].backward()
    n_none = sum(1 for p in list(pclf.parameters()) + list(hdc.parameters())
                 + list(dec.parameters()) if p.grad is None)
    print(f"params without grad (want 0): {n_none}", flush=True)
    assert n_none == 0
    n_params = (sum(p.numel() for p in pclf.parameters())
                + sum(p.numel() for p in hdc.parameters())
                + sum(p.numel() for p in dec.parameters())
                + sum(p.numel() for p in loss_fn.parameters()))

    subprocess.run(["./DIAG-code/sync.sh", "sha"], capture_output=True, check=True)
    sha = Path(".code_sha").read_text().strip()
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("diag-env-bringup")
    with mlflow.start_run(run_name="smoke-train-eq1617"):
        mlflow.set_tag("code_sha", sha)
        mlflow.set_tag("host", "local-cpu")
        mlflow.set_tag("node", "train-eq1617")
        mlflow.log_metric("loss_total", float(out["total"]))
        mlflow.log_metric("n_params", float(n_params))
    print(f"mlflow run ok code_sha={sha}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
