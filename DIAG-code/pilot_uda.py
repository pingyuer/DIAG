"""G2 CardiacUDA pilot: zero-shot ds-full ckpt, prompt-free, sparse labels.

Gate G0 passed (processed + metadata exist). G1: prompt-free, no GT/box in
forward; score only on labeled frames (label_valid from metadata label_indices).
128px inputs -> ContentAnchor handles any H,W (F_tf=H/2). Reports Dice/HD95 +
delta vs CAMUS baseline. No mDice (single domain, 000 sec.3).
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
from diag_metrics import dice_score, hd95
from diag.anchoring import ContentAnchor
from diag.decoder import CandidateDecoder
from diag.ds_head import DsHead
from diag.hdc import HDC
from diag.pclf import PCLF

MLFLOW_URI = "http://172.16.240.77:5000"
REMOTE_IMG = "root@172.16.240.188:/input3/processed/cardiacuda_a4c_lv_png128_10f/test"
LOCAL = Path("outputs/cardiacuda_pilot")
CKPT = Path(os.environ.get("DIAG_CKPT", "outputs/remote-31035/camus_train/dsfull_best.pt"))
N_CLIP = int(os.environ.get("DIAG_NCLIP", "29"))


def fetch():
    import subprocess
    LOCAL.mkdir(parents=True, exist_ok=True)
    if len(list(LOCAL.glob("*"))) >= N_CLIP:
        return
    subprocess.run(f"scp -o BatchMode=yes -q -P 32237 -r {REMOTE_IMG}/img {REMOTE_IMG}/label {REMOTE_IMG}/metadata {LOCAL}/",
                   shell=True, check=True)
    print(f"fetched -> {LOCAL}", flush=True)


def main():
    import subprocess
    torch.manual_seed(0)
    fetch()
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    print(f"ckpt ep={ckpt['epoch']} val={ckpt['val_dice']:.4f} (CAMUS-trained, zero-shot)", flush=True)
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

    clips = sorted((LOCAL / "img").iterdir())[:N_CLIP]
    print(f"clips={len(clips)}", flush=True)
    d_all, h_all, n_lab = [], [], 0
    for cd in clips:
        frames = sorted(cd.glob("*.png"))[:10]
        meta = json.loads((LOCAL / "metadata" / (cd.name + ".json")).read_text())
        lab_idx = set(meta.get("label_indices", list(range(10))))
        lb_dir = LOCAL / "label" / cd.name
        x = torch.stack([torch.from_numpy(np.array(Image.open(f))).float().div(255) for f in frames]).unsqueeze(1).unsqueeze(1)
        with torch.no_grad():
            feats = [anchor(f) for f in x]
            f_tf = torch.stack([o["F_tf"] for o in feats])
            f_tc = torch.stack([o["F_tc"] for o in feats])
            ds_vec = ds_head(f_tf)
            pf = pclf.forward(f_tf, f_tc, ds_vec[0])
            ho = hdc(f_tf, pf["fine"]["states"], pf["coarse"]["states"])
            do = dec(ho["F_e"], ho["P"], ho["Q_e"])
            sel = do["quality"].argmax(-1).reshape(-1)
            prob = torch.sigmoid(do["masks"]).reshape(-1, 3, 128, 128)
            best = prob[torch.arange(sel.shape[0]), sel]
            pred = (best > 0.5).float()  # (10,128,128) at 128 (F/2 of 128? decoder 2x of 64)
        # decoder out = 2x F_tf spatial (64->128 for 128px input): match label size
        for i in sorted(lab_idx):
            lf = lb_dir / f"{i:04d}.png"
            if not lf.exists():
                continue
            gt = (torch.from_numpy(np.array(Image.open(lf))).float() > 0.5).float()
            pr = pred[i]
            if pr.shape != gt.shape:
                pr = torch.nn.functional.interpolate(pr.unsqueeze(0).unsqueeze(0), size=tuple(gt.shape), mode="nearest").squeeze()
            d_all.append(float(dice_score(pr, gt).mean()))
            if gt.sum() > 0 and pr.sum() > 0:
                h_all.append(float(hd95(pr.unsqueeze(0), gt.unsqueeze(0)).mean()))
            n_lab += 1
    import numpy as np_
    print(f"labeled_frames={n_lab} dice={np_.mean(d_all):.4f} hd95={np_.mean(h_all):.2f} n={len(h_all)}", flush=True)
    print("CAMUS-delta: dice -0.9119 baseline; paper CardiacUDA DIAG .8092 / GDKVM .7279", flush=True)
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("diag-camus-train")
    with mlflow.start_run(run_name="cardiacuda-g2-pilot"):
        mlflow.set_tag("code_sha", ckpt.get("code_sha", "?"))
        mlflow.set_tag("node", "cross-domain-g2")
        mlflow.log_metric("dice", float(np_.mean(d_all)))
        mlflow.log_metric("hd95", float(np_.mean(h_all)))
        mlflow.log_metric("n_labeled", float(n_lab))
    print("mlflow g2 done", flush=True)


if __name__ == "__main__":
    main()
