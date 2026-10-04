"""Grid4 gamma-group diagnostics (006 grid4 + M10). Same lesson as grid3:
single ckpt single run is ONE data point, not a verdict. So: THREE ckpts
(grid1-s0/s1, dino-s) x full test50, report range, judge only on consensus.

1. Adjacent gamma-ratio CV (all channels, per spec).
2. Ordered-vs-shuffled FEATURE-level composition disruption Ratio
   (ordered recurrent modulation vs shuffled latent order; bar 1.05).
3. Latent hooks: cross-time swap + gaussian collapse, Dice drop.
"""
import json
import sys
from pathlib import Path

import mlflow
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "src")
sys.path.insert(0, "DIAG-code")
from diag_metrics import dice_score

from diag.anchoring import ContentAnchor
from diag.decoder import CandidateDecoder
from diag.dino_anchor import FrozenDinoAnchor
from diag.ds_head import DsHead
from diag.hdc import HDC
from diag.pclf import PCLF

MLFLOW_URI = "http://172.16.240.77:5000"
DATA = Path("outputs/camus_test_pull")
CKPTS = {
    "grid1-s0": ("outputs/grid1_s0_best.pt", "unext"),
    "grid1-s1": ("outputs/grid1_s1_best.pt", "unext"),
    "dino-s": ("outputs/dino_s_full_best.pt", "dinov2-small"),
}


def load_clip(pid: str):
    d = DATA / pid
    imgs = sorted((d / "img").glob("*.png"))[:10]
    gts = sorted((d / "gt_lv").glob("*.png"))[:10]
    x = torch.stack([torch.from_numpy(np.array(Image.open(f))).float().div(255) for f in imgs]).unsqueeze(1).unsqueeze(1)
    g = torch.stack([torch.from_numpy(np.array(Image.open(f))).float() for f in gts]).unsqueeze(1).unsqueeze(1)
    return x, (g > 0.5).float()


def build(ckpt_path: str, anchor_name: str):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    anchor = (ContentAnchor() if anchor_name == "unext"
              else FrozenDinoAnchor(anchor_name, out_channels=96)).eval()
    C = anchor.out_channels
    pclf = PCLF(C).eval()
    hdc = HDC(C, num_queries=4).eval()
    dec = CandidateDecoder(feat_dim=C, num_queries=4, num_candidates=3).eval()
    ds_head = DsHead(C).eval()
    anchor.load_state_dict(ckpt["anchor"])
    pclf.load_state_dict(ckpt["pclf"])
    hdc.load_state_dict(ckpt["hdc"])
    dec.load_state_dict(ckpt["dec"])
    if "ds_head" in ckpt:
        ds_head.load_state_dict(ckpt["ds_head"])
    return anchor, pclf, hdc, dec, ds_head, ckpt


def run_chain(anchor: torch.nn.Module, pclf: torch.nn.Module, hdc: torch.nn.Module,
                dec: torch.nn.Module, ds_head: torch.nn.Module, x: torch.Tensor,
                states_override: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, dict[str, dict[str, torch.Tensor]], dict[str, torch.Tensor]]:
    with torch.no_grad():
        feats = [anchor(f) for f in x]
        f_tf = torch.stack([o["F_tf"] for o in feats])
        f_tc = torch.stack([o["F_tc"] for o in feats])
        ds_vec = ds_head(f_tf)
        pf = pclf.forward(f_tf, f_tc, ds_vec[0])
        s_tf = states_override if states_override is not None else pf["fine"]["states"]
        ho = hdc(f_tf, s_tf, pf["coarse"]["states"])
        do = dec(ho["F_e"], ho["P"], ho["Q_e"])
        sel = do["quality"].argmax(-1).reshape(-1)
        prob = torch.sigmoid(do["masks"]).reshape(-1, 3, 256, 256)
        best = prob[torch.arange(sel.shape[0]), sel]
        return (best > 0.5).float().view(-1, 1, 1, 256, 256), pf, ho


def main():
    torch.manual_seed(0)
    split = json.loads(Path("/tmp/camus_split.json").read_text())
    test_ids = split["test_data"]
    summary = {}
    for cname, (cpath, aname) in CKPTS.items():
        anchor, pclf, hdc, dec, ds_head, ckpt = build(cpath, aname)
        print(f"== {cname} ep={ckpt['epoch']} val={ckpt['val_dice']:.4f}", flush=True)
        cvs, ratios, d_clean, d_swap, d_noise = [], [], [], [], []
        g = torch.Generator().manual_seed(0)
        for pid in test_ids:
            x, gt = load_clip(pid)
            pred, pf, ho = run_chain(anchor, pclf, hdc, dec, ds_head, x)
            d_clean.append(float(dice_score(pred, gt).mean()))
            # 1. gamma CV per clip (adjacent ratios over T)
            gm = ho["gamma"].detach()  # (T,B,C,H,W)
            rr = [(gm[k + 1] / gm[k].clamp(min=1e-8)) for k in range(gm.shape[0] - 1)]
            r = torch.stack(rr)
            cvs.append(float(r.std() / r.mean().clamp(min=1e-12)))
            # 2. ordered vs shuffled: shuffle STATE order, rerun HDC+dec
            perm = torch.randperm(10, generator=g)
            s_shuf = pf["fine"]["states"][perm]
            pred_s, _, _ = run_chain(anchor, pclf, hdc, dec, ds_head, x,
                                     states_override=s_shuf)
            d_shuf = float(dice_score(pred_s, gt).mean())
            base = max(d_clean[-1], 1e-9)
            ratios.append(d_clean[-1] / max(d_shuf, 1e-9))
            # 3a. cross-time swap: swap t0<->t9 states
            s_sw = pf["fine"]["states"].clone()
            s_sw[[0, 9]] = s_sw[[9, 0]]
            pred_sw, _, _ = run_chain(anchor, pclf, hdc, dec, ds_head, x,
                                      states_override=s_sw)
            d_swap.append(base - float(dice_score(pred_sw, gt).mean()))
            # 3b. gaussian collapse: states -> noise matched moments
            s_n = torch.randn(pf["fine"]["states"].shape, generator=g) * \
                pf["fine"]["states"].std() + pf["fine"]["states"].mean()
            pred_n, _, _ = run_chain(anchor, pclf, hdc, dec, ds_head, x,
                                     states_override=s_n)
            d_noise.append(base - float(dice_score(pred_n, gt).mean()))
        summary[cname] = {
            "cv_med": float(np.median(cvs)),
            "ratio_med": float(np.median(ratios)),
            "clean": float(np.mean(d_clean)),
            "swap_drop": float(np.mean(d_swap)),
            "noise_drop": float(np.mean(d_noise)),
        }
        s = summary[cname]
        print(f"{cname}: CV={s['cv_med']:.2e} ratio={s['ratio_med']:.3f} "
              f"clean={s['clean']:.4f} swap_drop={s['swap_drop']:.4f} "
              f"noise_drop={s['noise_drop']:.4f}", flush=True)
    cvs = [summary[k]["cv_med"] for k in summary]
    ratios = [summary[k]["ratio_med"] for k in summary]
    swaps = [summary[k]["swap_drop"] for k in summary]
    print(f"CONSENSUS(3 ckpts): CV range [{min(cvs):.2e},{max(cvs):.2e}] "
          f"ratio range [{min(ratios):.3f},{max(ratios):.3f}] "
          f"swap_drop range [{min(swaps):.4f},{max(swaps):.4f}]", flush=True)
    print("JUDGE(bars: ratio<1.05 or drop<0.01 -> H4-falsified-here): "
          f"ratio_min={min(ratios):.3f} swap_min={min(swaps):.4f}", flush=True)
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("diag-camus-train")
    with mlflow.start_run(run_name="grid4-gamma-3ckpt"):
        mlflow.set_tag("node", "grid4-gamma-group")
        mlflow.set_tag("phase", "grid4_gamma_group_diag")
        for k, s in summary.items():
            for m, v in s.items():
                mlflow.log_metric(f"{k}_{m}", v)
    print("mlflow grid4 done", flush=True)


if __name__ == "__main__":
    main()
