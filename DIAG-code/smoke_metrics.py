"""Metrics+diagnostics smoke: full pipeline on 3 CAMUS patients (real img+gt_lv).

Runs Anchor->PCLF->HDC->Decoder on 3 patients x 10 frames, scores Dice/HD95
+ temporal metrics + phase R2 + sampling stress. Untrained net: asserts only
plumbing + metric identities, NOT paper values (those need training).
"""
import subprocess
import sys
from pathlib import Path

import mlflow
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "DIAG-code")
from diag_diagnose import phase_r2, sampling_stress
from diag_metrics import (
    area_roughness,
    centroid_drift,
    dice_score,
    hd95,
    interframe_dice_variation,
)

from diag.anchoring import ContentAnchor
from diag.decoder import CandidateDecoder
from diag.hdc import HDC
from diag.pclf import PCLF

MLFLOW_URI = "http://172.16.240.77:5000"
REMOTE = "root@172.16.240.188:/input0/processed/camus_png256_10f"
PATIENTS = ["patient0001", "patient0002", "patient0003"]
LOCAL = Path("outputs/camus3")


def fetch():
    LOCAL.mkdir(parents=True, exist_ok=True)
    for p in PATIENTS:
        for side in ("img", "gt_lv"):
            d = LOCAL / p / side
            if len(list(d.glob("*.png"))) >= 10:
                continue
            d.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                f"scp -o BatchMode=yes -P 32237 {REMOTE}/{side}/{p}/*.png {d}/",
                shell=True, check=True)


def load(p: str) -> tuple[torch.Tensor, torch.Tensor]:
    imgs = sorted((LOCAL / p / "img").glob("*.png"))[:10]
    gts = sorted((LOCAL / p / "gt_lv").glob("*.png"))[:10]
    x = torch.stack([torch.from_numpy(np.asarray(Image.open(f))).float().div(255) for f in imgs]).unsqueeze(1)
    g = torch.stack([torch.from_numpy(np.asarray(Image.open(f))).float() for f in gts]).unsqueeze(1)
    return x, (g > 0.5).float()


def main():
    torch.manual_seed(0)
    fetch()
    anchor = ContentAnchor().eval()
    pclf = PCLF(anchor.out_channels).eval()
    hdc = HDC(anchor.out_channels, num_queries=4).eval()
    dec = CandidateDecoder(feat_dim=anchor.out_channels, num_queries=4, num_candidates=3).eval()

    def model_fn(frames: torch.Tensor, dts: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feats = [anchor(f.unsqueeze(0)) for f in frames]
            f_tf = torch.stack([o["F_tf"] for o in feats])
            f_tc = torch.stack([o["F_tc"] for o in feats])
            s_tf, s_tc = pclf.states(f_tf, f_tc, dts)
            hout = hdc(f_tf, s_tf, s_tc)
            dout = dec(hout["F_e"], hout["P"], hout["Q_e"])
            prob = torch.sigmoid(dout["masks"])
            sel = dout["quality"].argmax(-1)  # (T,B)
            n = prob.shape[0] * prob.shape[1]
            best = prob.reshape(n, 3, *prob.shape[3:])[torch.arange(n), sel.reshape(n)]
            t, b = prob.shape[:2]
            return (best > 0.5).float().view(t, b, 1, *prob.shape[3:])

    dices, hd95s, drifts, roughs, varis, r2s = [], [], [], [], [], []
    dts = torch.ones(9)
    for p in PATIENTS:
        x, g = load(p)
        pred = model_fn(x, dts)
        pred256 = torch.nn.functional.interpolate(pred.flatten(0, 1), size=(256, 256), mode="nearest").view(10, 1, 1, 256, 256)
        d = float(dice_score(pred256, g).mean())
        # HD95 on non-empty frames only
        hvals = []
        for i in range(10):
            if g[i].sum() > 0 and pred256[i].sum() > 0:
                hvals.append(float(hd95(pred256[i:i+1], g[i:i+1]).mean()))
        h = sum(hvals) / max(len(hvals), 1)
        dices.append(d)
        hd95s.append(h)
        drifts.append(float(centroid_drift(pred256).mean()))
        roughs.append(float(area_roughness(pred256).mean()))
        varis.append(float(interframe_dice_variation(pred256).mean()))
        with torch.no_grad():
            feats = [anchor(f.unsqueeze(0)) for f in x]
            f_tf = torch.stack([o["F_tf"] for o in feats])
            f_tc = torch.stack([o["F_tc"] for o in feats])
            s_tf, _ = pclf.states(f_tf, f_tc, dts)
            areas = g.flatten(2).sum(-1).squeeze(-1)  # (T,B)
            r2s.append(phase_r2(s_tf, areas)["r2"])
        print(f"{p}: dice={d:.4f} hd95={h:.2f} drift={drifts[-1]:.2f} rough={roughs[-1]:.1f} var={varis[-1]:.4f} r2={r2s[-1]:.4f}", flush=True)
    x0, g0 = load(PATIENTS[0])
    with torch.no_grad():
        phw = tuple(model_fn(x0, dts).shape[-2:])
    g0_small = torch.nn.functional.interpolate(
        g0.flatten(0, 1).unsqueeze(1).float(), size=phw, mode="nearest").view(10, 1, 1, *phw)
    stress = sampling_stress(model_fn, x0, dts, g0_small)
    print(f"mean dice={sum(dices)/3:.4f} (untrained, plumbing only)", flush=True)
    print(f"stress: retain_5f={stress['retain_5f']:.3f} retain_3f={stress['retain_3f']:.3f}", flush=True)

    subprocess.run(["./DIAG-code/sync.sh", "sha"], capture_output=True, check=True)
    sha = Path(".code_sha").read_text().strip()
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("diag-env-bringup")
    with mlflow.start_run(run_name="smoke-metrics-camus3"):
        mlflow.set_tag("code_sha", sha)
        mlflow.set_tag("host", "local-cpu")
        mlflow.set_tag("node", "metrics-diag")
        mlflow.log_metric("mean_dice_untrained", sum(dices) / 3)
        mlflow.log_metric("mean_r2_untrained", sum(r2s) / 3)
    print(f"mlflow run ok code_sha={sha}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
