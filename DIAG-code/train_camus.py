"""DIAG CAMUS full-chain training (remote, single GPU).

Why this config (not just defaults):
- AdamW lr=3e-4: conv+attention hybrid trunks train stably here (UNeXt-family
  practice); weight_decay=1e-2 regularizes the small-data regime (CAMUS 400
  train patients, tiny vs natural-image scale).
- ReduceLROnPlateau on val Dice (patience 5): echo Dice plateaus in steps as
  candidates specialize (WTA); step decay would cut lr mid-specialization.
- Labeled frames Omega = all 10 frames (CAMUS dense labels). No unlabeled
  handling needed for this dataset.
- dts = ones (no timestamp metadata in processed PNGs; recorded deviation).
  dt path already proven live in smoke; =1 reduces Euler to residual update.
- Batch = 2 clips (10x1x256x256): A30 24G fits easily; larger batch smooths
  WTA winner statistics (best-index histogram over N=T*B*clips samples).
- Inference selection = argmax quality; reported val Dice uses it (paper Eq.16).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import mlflow
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, "src")
sys.path.insert(0, "DIAG-code")
from diag_metrics import dice_score
from diag.anchoring import BACKBONE_RECORD, ContentAnchor
from diag.decoder import CandidateDecoder
from diag.hdc import HDC
from diag.losses import DiagLoss, DiagLossWeights
from diag.pclf import PCLF

DATA = Path("/input0/processed/camus_png256_10f")
OUT = Path("/root/DIAG/outputs/camus_train")
MLFLOW_URI = "http://172.16.240.77:5000"


def load_patient(p: str):
    imgs = sorted((DATA / "img" / p).glob("*.png"))[:10]
    gts = sorted((DATA / "gt_lv" / p).glob("*.png"))[:10]
    x = torch.stack([torch.from_numpy(__import__("numpy").asarray(Image.open(f))).float().div(255)
                     for f in imgs]).unsqueeze(1)
    g = torch.stack([torch.from_numpy(__import__("numpy").asarray(Image.open(f))).float()
                     for f in gts]).unsqueeze(1)
    return x, (g > 0.5).float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--subset", type=int, default=0, help="0=all train patients")
    ap.add_argument("--run-name", default="camus-fullchain")
    args = ap.parse_args()

    dev = torch.device("cuda:0")
    torch.manual_seed(0)
    split = json.loads((DATA / "camus_public_datasplit_20250706.json").read_text())
    train_ids = split["train_data"][: args.subset or None]
    val_ids = split["val_data"]
    print(f"train={len(train_ids)} val={len(val_ids)} epochs={args.epochs} batch={args.batch}", flush=True)

    anchor = ContentAnchor().to(dev)
    C = anchor.out_channels
    pclf = PCLF(C).to(dev)
    hdc = HDC(C, num_queries=4).to(dev)
    dec = CandidateDecoder(feat_dim=C, num_queries=4, num_candidates=3).to(dev)
    loss_fn = DiagLoss(DiagLossWeights(), state_dim=C).to(dev)
    params = list(anchor.parameters()) + list(pclf.parameters()) + list(hdc.parameters()) \
        + list(dec.parameters()) + list(loss_fn.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", patience=5, factor=0.5)

    def step(batch_ids: list[str], train: bool):
        if train:
            for m in (anchor, pclf, hdc, dec, loss_fn):
                m.train()
        else:
            for m in (anchor, pclf, hdc, dec, loss_fn):
                m.eval()
        tot, dsum, n = 0.0, 0.0, 0
        for p in batch_ids:
            x, g = load_patient(p)
            x = x.to(dev)
            g_small = F.interpolate(g.flatten(0, 1).unsqueeze(1), size=(256, 256), mode="nearest").view(10, 1, 1, 256, 256).to(dev)
            dts = torch.ones(9, device=dev)
            if train:
                opt.zero_grad()
            with torch.set_grad_enabled(train):
                feats = [anchor(f.unsqueeze(0)) for f in x]
                f_tf = torch.stack([o["F_tf"] for o in feats])
                f_tc = torch.stack([o["F_tc"] for o in feats])
                pf = pclf.forward(f_tf, f_tc, dts)
                s_tf, s_tc = pf["fine"]["states"], pf["coarse"]["states"]
                ho = hdc(f_tf, s_tf, s_tc)
                do = dec(ho["F_e"], ho["P"], ho["Q_e"])
                lab = torch.ones(10, 1, dtype=torch.bool, device=dev)
                out = loss_fn(do["masks"], do["quality"], g_small, lab, s_tf, pf["fine"]["obs"], dts)
                if train:
                    out["total"].backward()
                    opt.step()
            with torch.no_grad():
                sel = do["quality"].argmax(-1)
                prob = torch.sigmoid(do["masks"])
                idx = sel.reshape(-1)
                best = prob.reshape(-1, 3, 256, 256)[torch.arange(idx.shape[0]), idx]
                dsum += float(dice_score((best > 0.5).float().view(10, 1, 1, 256, 256), g_small).mean())
            tot += float(out["total"])
            n += 1
        return tot / n, dsum / n

    code_sha = Path("/root/DIAG/.code_sha").read_text().strip()
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("diag-camus-train")
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with mlflow.start_run(run_name=args.run_name) as run:
        mlflow.set_tag("code_sha", code_sha)
        mlflow.set_tag("node", "camus-fullchain-train")
        mlflow.set_tag("backbone", BACKBONE_RECORD["name"] + "-random-init-DEVIATION")
        mlflow.set_tag("lambda", "placeholder-supplement-pending-DEVIATION")
        mlflow.set_tag("dts", "ones-no-timestamp-DEVIATION")
        mlflow.log_param("epochs", args.epochs)
        mlflow.log_param("batch", args.batch)
        mlflow.log_param("lr", args.lr)
        mlflow.log_param("train_n", len(train_ids))
        best_val = 0.0
        for ep in range(args.epochs):
            tl_acc, td_acc, nb = 0.0, 0.0, 0
            for i in range(0, len(train_ids), args.batch):
                l, d = step(train_ids[i:i + args.batch], True)
                tl_acc += l
                td_acc += d
                nb += 1
            tl, td = tl_acc / nb, td_acc / nb
            vl, vd = step(val_ids, False)
            sched.step(vd)
            mlflow.log_metric("train_loss", tl, step=ep)
            mlflow.log_metric("train_dice", td, step=ep)
            mlflow.log_metric("val_loss", vl, step=ep)
            mlflow.log_metric("val_dice", vd, step=ep)
            mlflow.log_metric("lr", opt.param_groups[0]["lr"], step=ep)
            print(f"ep{ep:03d} train_loss={tl:.4f} train_dice={td:.4f} val_loss={vl:.4f} val_dice={vd:.4f} "
                  f"lr={opt.param_groups[0]['lr']:.1e} t={time.time()-t0:.0f}s", flush=True)
            ckpt = {"epoch": ep, "anchor": anchor.state_dict(), "pclf": pclf.state_dict(),
                    "hdc": hdc.state_dict(), "dec": dec.state_dict(), "opt": opt.state_dict(),
                    "val_dice": vd, "code_sha": code_sha}
            torch.save(ckpt, OUT / "last.pt")
            if vd > best_val:
                best_val = vd
                torch.save(ckpt, OUT / "best.pt")
                mlflow.log_metric("best_val_dice", best_val, step=ep)
        print(f"DONE best_val_dice={best_val:.4f} ckpt={OUT}/best.pt run={run.info.run_id}", flush=True)


if __name__ == "__main__":
    main()
