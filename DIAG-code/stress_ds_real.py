"""Grid3 ds real-interval stress: TRUE ds-head path vs fixed dt=1.0 (006 grid3).

Voids P4-synthetic (scalar jitter bypassed ds head; self-limited per eval_004).
Here every scenario runs BOTH rails on the SAME ckpt:
  rail A (clock): frames -> anchor -> ds_head -> dt_hat -> PCLF -> HDC -> dec
  rail B (fixed): same frames, dt = ones (ds ignored)
Scenarios (test50 full, patient-avg): stride2, stride3, rand-drop-50%, static
(zero-motion: repeat frame0 x10 -> ds should floor, Dice vs self not vs GT).
Verdict: retention gap |A-B| <= 0.3pt OR dt_hat stride-change < 5% -> clock
falsified; dt floor-touch > 10% -> alarm.
"""
import json
import os
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
from diag.ds_head import DsHead
from diag.hdc import HDC
from diag.pclf import PCLF

MLFLOW_URI = "http://172.16.240.77:5000"
DATA = Path("outputs/camus_test_pull")
CKPT = Path(os.environ.get("DIAG_CKPT", "outputs/grid1_s0_best.pt"))


def load_clip(pid: str):
    d = DATA / pid
    imgs = sorted((d / "img").glob("*.png"))[:10]
    gts = sorted((d / "gt_lv").glob("*.png"))[:10]
    x = torch.stack([torch.from_numpy(np.array(Image.open(f))).float().div(255) for f in imgs]).unsqueeze(1).unsqueeze(1)
    g = torch.stack([torch.from_numpy(np.array(Image.open(f))).float() for f in gts]).unsqueeze(1).unsqueeze(1)
    return x, (g > 0.5).float()


def infer(anchor, pclf, hdc, dec, ds_head, x, use_ds: bool):
    with torch.no_grad():
        feats = [anchor(f) for f in x]
        f_tf = torch.stack([o["F_tf"] for o in feats])
        f_tc = torch.stack([o["F_tc"] for o in feats])
        ds_vec = ds_head(f_tf)
        dts = ds_vec[0] if use_ds else torch.ones(f_tf.shape[0] - 1)
        pf = pclf.forward(f_tf, f_tc, dts)
        ho = hdc(f_tf, pf["fine"]["states"], pf["coarse"]["states"])
        do = dec(ho["F_e"], ho["P"], ho["Q_e"])
        sel = do["quality"].argmax(-1).reshape(-1)
        prob = torch.sigmoid(do["masks"]).reshape(-1, 3, 256, 256)
        best = prob[torch.arange(sel.shape[0]), sel]
        pred = (best > 0.5).float().view(-1, 1, 1, 256, 256)
        return pred, ds_vec


def main():
    torch.manual_seed(0)
    split = json.loads(Path("/tmp/camus_split.json").read_text())
    test_ids = split["test_data"]
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    print(f"ckpt ep={ckpt['epoch']} val={ckpt['val_dice']:.4f} run={ckpt.get('code_sha','?')[:8]}", flush=True)
    anchor = ContentAnchor().eval()
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
    g = torch.Generator().manual_seed(0)
    scenarios = {
        "full": list(range(10)),
        "stride2": list(range(0, 10, 2)),
        "stride3": list(range(0, 10, 3)),
        "drop50": sorted(torch.randperm(10, generator=g)[:5].tolist()),
    }
    res = {}
    for sname, idx in scenarios.items():
        dA, dB, ds_s, fl = [], [], [], []
        for pid in test_ids:
            x, gt = load_clip(pid)
            xs, gs = x[idx], gt[idx]
            predA, ds_vec = infer(anchor, pclf, hdc, dec, ds_head, xs, True)
            predB, _ = infer(anchor, pclf, hdc, dec, ds_head, xs, False)
            dA.append(float(dice_score(predA, gs).mean()))
            dB.append(float(dice_score(predB, gs).mean()))
            ds_s.append(ds_vec.detach())
            fl.append(float((ds_vec <= 1.5e-3).float().mean()))
        res[sname] = (float(np.mean(dA)), float(np.mean(dB)),
                      torch.cat(ds_s).mean().item(), float(np.mean(fl)))
        print(f"{sname}: railA={res[sname][0]:.4f} railB={res[sname][1]:.4f} "
              f"gap={res[sname][0]-res[sname][1]:+.4f} ds_mean={res[sname][2]:.4f} floor={res[sname][3]:.3f}", flush=True)
    # static: repeat frame0
    dA_s, dB_s, ds_s0, fl0 = [], [], [], []
    for pid in test_ids:
        x, gt = load_clip(pid)
        xs = x[[0]].expand(10, -1, -1, -1, -1).clone()
        predA, ds_vec = infer(anchor, pclf, hdc, dec, ds_head, xs, True)
        predB, _ = infer(anchor, pclf, hdc, dec, ds_head, xs, False)
        dA_s.append(float(dice_score(predA, gt).mean()))
        dB_s.append(float(dice_score(predB, gt).mean()))
        ds_s0.append(ds_vec.detach())
        fl0.append(float((ds_vec <= 1.5e-3).float().mean()))
    print(f"static: railA={np.mean(dA_s):.4f} railB={np.mean(dB_s):.4f} "
          f"ds_mean={torch.cat(ds_s0).mean():.4f} floor={np.mean(fl0):.3f} "
          f"(Dice vs own GT; ds should floor on zero motion)", flush=True)
    # verdict
    gaps = [abs(res[k][0] - res[k][1]) for k in ("stride2", "stride3", "drop50")]
    full_ds = res["full"][2]
    str2_ds = res["stride2"][2]
    rel = abs(str2_ds - full_ds) / max(full_ds, 1e-9)
    print(f"VERDICT max_gap={max(gaps):.4f} (falsify if <=0.003) "
          f"ds_stride_change={rel:.3f} (falsify if <0.05)", flush=True)
    try:
        sha = Path(".code_sha").read_text().strip()
    except OSError:
        sha = ckpt.get("code_sha", "?")
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("diag-camus-train")
    with mlflow.start_run(run_name="grid3-ds-real"):
        mlflow.set_tag("code_sha", sha)
        mlflow.set_tag("node", "grid3-ds-real-stress")
        mlflow.set_tag("phase", "grid3_ds_real_stress")
        mlflow.set_tag("ckpt", str(CKPT))
        for k in scenarios:
            mlflow.log_metric(f"railA_{k}", res[k][0])
            mlflow.log_metric(f"railB_{k}", res[k][1])
    print("mlflow grid3 done", flush=True)


if __name__ == "__main__":
    main()
