"""P1-P6 ds stress: offline perturbations on ONE ckpt (004 sec.1).

Same chain as eval_test (Anchor->PCLF(ds)->HDC->Decoder), ds-full ckpt,
test split. No retraining. Reports retention vs full + ds/K diagnostics.

P1 5f / P2 3f / P3 70% drop / P4 40% jitter / P5 ds floor check / P6 K drift.
Limit (004): PNGs carry no wall-clock; P4 jitter is a synthetic multiplier.
"""
import os
import sys
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn.functional as F
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
CKPT = Path(os.environ.get("DIAG_CKPT", "outputs/remote-31035/camus_train/dsfull_best.pt"))
N_PAT = int(os.environ.get("DIAG_NPAT", "15"))


def load_clip(pid: str):
    d = DATA / pid
    imgs = sorted((d / "img").glob("*.png"))[:10]
    gts = sorted((d / "gt_lv").glob("*.png"))[:10]
    x = torch.stack([torch.from_numpy(np.array(Image.open(f))).float().div(255) for f in imgs]).unsqueeze(1).unsqueeze(1)
    g = torch.stack([torch.from_numpy(np.array(Image.open(f))).float() for f in gts]).unsqueeze(1).unsqueeze(1)
    return x, (g > 0.5).float()


def main():
    import json
    import subprocess
    torch.manual_seed(0)
    split = json.loads(Path("/tmp/camus_split.json").read_text())
    test_ids = split["test_data"][:N_PAT]
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    print(f"ckpt ep={ckpt['epoch']} val={ckpt['val_dice']:.4f} sha={ckpt.get('code_sha','?')[:8]}", flush=True)
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
    else:
        print("WARN: ckpt lacks ds_head (pre-fix run); ds at init 1.0", flush=True)

    def infer(x, dts):
        with torch.no_grad():
            feats = [anchor(f) for f in x]
            f_tf = torch.stack([o["F_tf"] for o in feats])
            f_tc = torch.stack([o["F_tc"] for o in feats])
            ds_vec = ds_head(f_tf)
            if dts is None:
                dts = ds_vec[0]
            pf = pclf.forward(f_tf, f_tc, dts)
            ho = hdc(f_tf, pf["fine"]["states"], pf["coarse"]["states"])
            do = dec(ho["F_e"], ho["P"], ho["Q_e"])
            sel = do["quality"].argmax(-1).reshape(-1)
            prob = torch.sigmoid(do["masks"]).reshape(-1, 3, 256, 256)
            best = prob[torch.arange(sel.shape[0]), sel]
            return (best > 0.5).float().view(-1, 1, 1, 256, 256), ds_vec, pf

    g = torch.Generator().manual_seed(0)
    acc = {"full": [], "p1": [], "p2": [], "p3": [], "p4": []}
    ds_floor, k_full, k_p4 = [], [], []
    for pid in test_ids:
        x, gt = load_clip(pid)
        dts = torch.ones(9)
        pred, ds_vec, pf = infer(x, None)
        base = float(dice_score(pred, gt).mean())
        acc["full"].append(base)
        ds_floor.append(float(ds_vec.min()))
        k_full.append(float(pf["fine"]["gates"].mean()))
        # P1/P2 uniform subsample
        for key, idx in (("p1", torch.linspace(0, 9, 5).round().long()),
                         ("p2", torch.linspace(0, 9, 3).round().long())):
            xs, gs = x[idx], gt[idx]
            dd = dts[idx[1:] - 1] if False else torch.ones(len(idx) - 1)
            pr, _, _ = infer(xs, None)
            acc[key].append(float(dice_score(pr, gs).mean()) / max(base, 1e-9))
        # P3 70% drop (keep 3)
        keep = sorted(torch.randperm(10, generator=g)[:3].tolist())
        pr3, _, _ = infer(x[keep], None)
        acc["p3"].append(float(dice_score(pr3, gt[keep]).mean()) / max(base, 1e-9))
        # P4 jitter xU(0.6,1.4)
        jit = torch.ones(9) * (1 + (torch.rand(9, generator=g) - 0.5) * 0.8)
        with torch.no_grad():
            feats = [anchor(f) for f in x]
            f_tf = torch.stack([o["F_tf"] for o in feats])
            f_tc = torch.stack([o["F_tc"] for o in feats])
            pfj = pclf.forward(f_tf, f_tc, jit)
            hoj = hdc(f_tf, pfj["fine"]["states"], pfj["coarse"]["states"])
            doj = dec(hoj["F_e"], hoj["P"], hoj["Q_e"])
            sel = doj["quality"].argmax(-1).reshape(-1)
            prob = torch.sigmoid(doj["masks"]).reshape(-1, 3, 256, 256)
            pr4 = (prob[torch.arange(sel.shape[0]), sel] > 0.5).float().view(10, 1, 1, 256, 256)
        acc["p4"].append(float(dice_score(pr4, gt).mean()) / max(base, 1e-9))
        k_p4.append(float(pfj["fine"]["gates"].mean()))
    import numpy as np_
    print(f"n={len(test_ids)} full_dice={np_.mean(acc['full']):.4f}", flush=True)
    for k in ("p1", "p2", "p3", "p4"):
        print(f"retain_{k}={np_.mean(acc[k]):.4f} (paper {[0.989, 0.959, 0.970, 0.975][['p1','p2','p3','p4'].index(k)]})", flush=True)
    floor_ratio = float((torch.tensor(ds_floor) <= 1.5e-3).float().mean())
    print(f"ds_min_mean={np_.mean(ds_floor):.4f} floor_ratio={floor_ratio:.3f} (P5 alarm >0.10)", flush=True)
    print(f"k_full={np_.mean(k_full):.3f} k_jitter={np_.mean(k_p4):.3f} (P6 drift)", flush=True)

    try:
        sha = Path(".code_sha").read_text().strip()
    except OSError:
        sha = ckpt.get("code_sha", "?")
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("diag-camus-train")
    with mlflow.start_run(run_name="ds-stress-p1p6"):
        mlflow.set_tag("code_sha", sha)
        mlflow.set_tag("node", "ds-stress")
        mlflow.set_tag("ckpt", str(CKPT))
        for k in ("p1", "p2", "p3", "p4"):
            mlflow.log_metric(f"retain_{k}", float(np_.mean(acc[k])))
        mlflow.log_metric("ds_floor_ratio", floor_ratio)
        mlflow.log_metric("k_drift", float(np_.mean(k_p4) - np_.mean(k_full)))
    print("mlflow ds-stress done", flush=True)


if __name__ == "__main__":
    main()
