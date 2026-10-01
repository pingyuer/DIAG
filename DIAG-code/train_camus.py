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

W_THRESHOLDS = {  # O1-O10 watch table (002 sec.2); written as run tags
    "watch/gamma_sat": 0.3,
    "watch/k_lo": 0.05,
    "watch/k_hi": 0.95,
    "watch/best_mono": 0.95,
    "watch/val_stall": 0.001,
    "watch/overfit_eps": 3,
    "watch/hd95_tol": 1.10,  # x paper 5.43
    "watch/cv_tol": 1e-4,  # 100x paper 1e-6
    "watch/ds_min": 0.05,  # ds collapse alarm (003 sec.5.4)
}

sys.path.insert(0, "src")
sys.path.insert(0, "DIAG-code")
from diag_metrics import dice_score
from diag.anchoring import BACKBONE_RECORD, ContentAnchor
from diag.decoder import CandidateDecoder
from diag.hdc import HDC
from diag.losses import DiagLoss, DiagLossWeights
from diag.ds_head import DsHead
from diag.pclf import PCLF

DATA = Path("/input0/processed/camus_png256_10f")
OUT = Path("/root/DIAG/outputs/camus_train")
MLFLOW_URI = "http://172.16.240.77:5000"


def load_patient(p: str):
    imgs = sorted((DATA / "img" / p).glob("*.png"))[:10]
    gts = sorted((DATA / "gt_lv" / p).glob("*.png"))[:10]
    x = torch.stack([torch.from_numpy(__import__("numpy").asarray(Image.open(f))).float().div(255)
                     for f in imgs]).unsqueeze(1).unsqueeze(1)  # (T,1,1,H,W)
    g = torch.stack([torch.from_numpy(__import__("numpy").asarray(Image.open(f))).float()
                     for f in gts]).unsqueeze(1).unsqueeze(1)
    return x, (g > 0.5).float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--subset", type=int, default=0, help="0=all train patients")
    ap.add_argument("--run-name", default="camus-fullchain")
    ap.add_argument("--w-step", type=int, default=50, help="W-step log every N steps (002 sec.1)")
    ap.add_argument("--diag-every", type=int, default=5, help="W-event diagnose snapshot every K epochs (0=off)")
    ap.add_argument("--frame-only", action="store_true",
                    help="H1 baseline: bypass PCLF/HDC, decode F_tf directly (no dynamics)")
    ap.add_argument("--svf", action="store_true", help="SVF warp branch ON (004 T1-T3)")
    ap.add_argument("--ss-steps", type=int, default=6)
    ap.add_argument("--max-disp", type=float, default=0.05)
    ap.add_argument("--svf-smooth", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ds-lr", type=float, default=1e-4, help="005 sec.1.2 ds head own lr (0=keep no_grad frozen)")
    ap.add_argument("--norm", action="store_true", help="005 sec.1.1 loss running-mean norm")
    ap.add_argument("--l-boundary", type=float, default=0.0, help="005 sec.1.4 boundary loss weight")
    ap.add_argument("--l-ce", type=float, default=1.0)
    ap.add_argument("--l-dice", type=float, default=1.0)
    ap.add_argument("--l-rec", type=float, default=0.1)
    ap.add_argument("--l-smooth", type=float, default=0.01)
    ap.add_argument("--l-flow", type=float, default=0.1)
    ap.add_argument("--l-iou", type=float, default=0.5)
    args = ap.parse_args()

    dev = torch.device("cuda:0")
    torch.manual_seed(args.seed)
    split = json.loads((DATA / "camus_public_datasplit_20250706.json").read_text())
    train_ids = split["train_data"][: args.subset or None]
    val_ids = split["val_data"]
    print(f"train={len(train_ids)} val={len(val_ids)} epochs={args.epochs} batch={args.batch}", flush=True)

    anchor = ContentAnchor().to(dev)
    C = anchor.out_channels
    pclf = PCLF(C).to(dev)
    ds_head = DsHead(C).to(dev)
    hdc = HDC(C, num_queries=4).to(dev)
    dec = CandidateDecoder(feat_dim=C, num_queries=4, num_candidates=3).to(dev)
    loss_fn = DiagLoss(DiagLossWeights(ce=args.l_ce, dice=args.l_dice, rec=args.l_rec,
                                         smooth=args.l_smooth, flow=args.l_flow,
                                         iou=args.l_iou,
                                         boundary=args.l_boundary), state_dim=C, norm=args.norm).to(dev)
    # SVF REMOVED from chain (003 fallback, grid 18/18 negative 2026-10-01):
    # val_dice 0.65-0.73 all-grid, HD95 51-68 collapsed, delta>0 but hurts.
    # svf.py stays as diagnose tool; decoder phi=None default pass-through.
    svf_head = None
    params = list(anchor.parameters()) + list(pclf.parameters()) + list(hdc.parameters()) \
        + list(dec.parameters()) + list(loss_fn.parameters()) + list(ds_head.parameters())
    ds_params = list(ds_head.parameters())
    main_params = [p for p in params if not any(p is q for q in ds_params)]
    opt = torch.optim.AdamW(main_params, lr=args.lr, weight_decay=1e-2)
    opt_ds = torch.optim.AdamW(ds_params, lr=args.ds_lr, weight_decay=0.0) if args.ds_lr > 0 else None
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", patience=5, factor=0.5)

    W = {"n": 0, "ce": 0.0, "dice": 0.0, "rec": 0.0, "smooth": 0.0, "flow": 0.0,
         "iou": 0.0, "boundary": 0.0, "k": 0.0, "gamma": 0.0, "gsat": 0.0,
         "best": [0, 0, 0], "alpha_c": 0.0, "alpha_f": 0.0, "gn_pre": 0.0, "gn_post": 0.0,
         "ds_mean": 0.0, "ds_min": 0.0, "svf_delta": 0.0, "svf_smooth": 0.0}

    def w_log(suffix: str, gstep: int) -> None:
        # W-step window (002 sec.1): six loss splits + K/gamma/best/alpha + grad norms
        n = max(W["n"], 1)
        mlflow.log_metric(f"wstep/loss_total{suffix}", W["ce"] / n + W["dice"] / n, step=gstep)
        for k in ("ce", "dice", "rec", "smooth", "flow", "iou", "boundary"):
            mlflow.log_metric(f"wstep/loss_{k}{suffix}", W[k] / n, step=gstep)
        mlflow.log_metric(f"wstep/k_mean{suffix}", W["k"] / n, step=gstep)
        mlflow.log_metric(f"wstep/gamma_mean{suffix}", W["gamma"] / n, step=gstep)
        mlflow.log_metric(f"wstep/gamma_sat{suffix}", W["gsat"] / n, step=gstep)
        mlflow.log_metric(f"wstep/alpha_c{suffix}", W["alpha_c"] / n, step=gstep)
        mlflow.log_metric(f"wstep/alpha_f{suffix}", W["alpha_f"] / n, step=gstep)
        mlflow.log_metric(f"wstep/grad_pre{suffix}", W["gn_pre"] / n, step=gstep)
        mlflow.log_metric(f"wstep/grad_post{suffix}", W["gn_post"] / n, step=gstep)
        mlflow.log_metric(f"wstep/ds_mean{suffix}", W.get("ds_mean", 0.0) / n, step=gstep)
        mlflow.log_metric(f"wstep/ds_min{suffix}", W.get("ds_min", 0.0) / n, step=gstep)
        mlflow.log_metric(f"wstep/svf_delta{suffix}", W.get("svf_delta", 0.0) / n, step=gstep)
        mlflow.log_metric(f"wstep/svf_smooth{suffix}", W.get("svf_smooth", 0.0) / n, step=gstep)
        btot = max(sum(W["best"]), 1)
        for j in range(3):
            mlflow.log_metric(f"wstep/best_j{j}{suffix}", W["best"][j] / btot, step=gstep)

    def w_reset() -> None:
        for k in ("ce", "dice", "rec", "smooth", "flow", "iou", "boundary", "k", "gamma", "gsat",
                  "alpha_c", "alpha_f", "gn_pre", "gn_post", "ds_mean", "ds_min",
                  "svf_delta", "svf_smooth"):
            W[k] = 0.0
        W["n"] = 0
        W["best"] = [0, 0, 0]

    def step(batch_ids: list[str], train: bool, gstep: list[int] | None = None,
             suffix: str = ""):
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
            g_small = F.interpolate(g.flatten(0, 1), size=(256, 256), mode="nearest").view(10, 1, 1, 256, 256).to(dev)
            if train:
                opt.zero_grad()
            with torch.set_grad_enabled(train):
                # batched anchor: 10 frames one call (~4x faster than per-frame loop)
                xb = x.squeeze(1)  # (T,1,H,W) BCHW batch-of-frames
                aout = anchor(xb)
                f_tf = aout["F_tf"].unsqueeze(1)  # (T,1,C,H/2,W/2)
                f_tc = aout["F_tc"].unsqueeze(1)
                if opt_ds is not None:
                    ds_head.train()
                    ds_vec = ds_head(f_tf)  # detached inside: backbone cut, ds learns
                    dts = ds_vec[0]
                else:
                    with torch.no_grad():
                        ds_vec = ds_head(f_tf)
                    dts = ds_vec[0]
                if args.frame_only:
                    # H1 baseline: no dynamics; decode content features directly.
                    # Q_e zeros keep decoder signature; quality still learns selection.
                    t_, b_ = f_tf.shape[:2]
                    zq = torch.zeros(t_, b_, 4, f_tf.shape[2], device=dev)
                    do = dec(f_tf, f_tf, zq)
                    pf = None
                    s_tf = f_tf  # rec/smooth/flow targets degrade to identity
                    obs_f = f_tf
                    ho = {"gamma": torch.full_like(f_tf[:, :, :1], 0.88),
                          "alpha_c": torch.tensor(0.0), "alpha_f": torch.tensor(0.0)}
                else:
                    pf = pclf.forward(f_tf, f_tc, dts)
                    s_tf, s_tc = pf["fine"]["states"], pf["coarse"]["states"]
                    ho = hdc(f_tf, s_tf, s_tc, record_gamma=True)
                    if False:  # SVF deleted, see note above
                        raise RuntimeError("unreachable")
                    else:
                        do = dec(ho["F_e"], ho["P"], ho["Q_e"])
                        sv_smooth_extra = None
                        sv_delta = 0.0
                    obs_f = pf["fine"]["obs"]
                lab = torch.ones(10, 1, dtype=torch.bool, device=dev)
                out = loss_fn(do["masks"], do["quality"], g_small, lab, s_tf, obs_f, dts)
                if sv_smooth_extra is not None:
                    out = dict(out)
                    out["total"] = out["total"] + sv_smooth_extra
                if train:
                    if opt_ds is not None:
                        opt_ds.zero_grad()
                    out["total"].backward()
                    if opt_ds is not None:
                        opt_ds.step()
                    gn_pre = torch.nn.utils.clip_grad_norm_(params, 1.0)
                    gn_post = sum(p.grad.norm().item() ** 2 for p in params if p.grad is not None) ** 0.5
                    opt.step()
                    # W-step accumulators (train only)
                    W["n"] += 1
                    for k in ("ce", "dice", "rec", "smooth", "flow", "iou", "boundary"):
                        W[k] += float(out[k])
                    kf = pf["fine"]["gates"].detach() if pf is not None else None
                    W["k"] += float(kf.mean()) if kf is not None else 0.0
                    gm = ho["gamma"].detach()
                    W["gamma"] += float(gm.mean())
                    W["gsat"] += float((gm > 0.95).float().mean())
                    W["alpha_c"] += float(ho["alpha_c"])
                    W["alpha_f"] += float(ho["alpha_f"])
                    W["gn_pre"] += float(gn_pre)
                    W["gn_post"] += float(gn_post)
                    for j in out["best"].tolist():
                        W["best"][int(j)] += 1
                    W["ds_mean"] = W.get("ds_mean", 0.0) + float(ds_vec.mean())
                    W["ds_min"] = W.get("ds_min", 0.0) + float(ds_vec.min())
                    W["svf_delta"] = W.get("svf_delta", 0.0) + sv_delta
                    W["svf_smooth"] = W.get("svf_smooth", 0.0) + float(
                        sv_smooth_extra.detach() if sv_smooth_extra is not None
                        else torch.zeros(()))
                    if gstep is not None and W["n"] % max(args.w_step, 1) == 0:
                        gstep[0] += 1
                        w_log(suffix, gstep[0])
                        w_reset()
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
        mlflow.set_tag("node", "frame-only-baseline" if args.frame_only else "camus-fullchain-train")
        mlflow.set_tag("norm", str(bool(args.norm)))
        mlflow.set_tag("backbone", BACKBONE_RECORD["name"] + "-random-init-DEVIATION")
        mlflow.set_tag("lambda", "placeholder-supplement-pending-DEVIATION")
        mlflow.set_tag("dts", "ds-head-learned-DEVIATION-fallback-ones")
        for k, v in W_THRESHOLDS.items():
            mlflow.set_tag(k, str(v))
        for k, v in vars(args).items():
            mlflow.log_param(k, v)
        mlflow.log_param("train_n", len(train_ids))
        mlflow.log_param("val_n", len(val_ids))
        best_val = 0.0
        gstep = [0]
        from diag_metrics import (area_roughness, centroid_drift, hd95,
                                  interframe_dice_variation)
        vids = val_ids[:8]  # W-epoch temporal/HD95 subset (cost control)
        for ep in range(args.epochs):
            tl_acc, td_acc, nb = 0.0, 0.0, 0
            for i in range(0, len(train_ids), args.batch):
                l, d = step(train_ids[i:i + args.batch], True, gstep, suffix="")
                tl_acc += l
                td_acc += d
                nb += 1
            tl, td = tl_acc / nb, td_acc / nb
            vl, vd = step(val_ids, False)
            sched.step(vd)
            # W-epoch: loss-epoch means + HD95/temporal/ED-ES-Mean on val subset
            hd_list, dr_list, rg_list, va_list, ed_list, es_list = [], [], [], [], [], []
            with torch.no_grad():
                for m in (anchor, pclf, hdc, dec):
                    m.eval()
                for pid in vids:
                    xv, gv = load_patient(pid)
                    xv = xv.to(dev)
                    gs = F.interpolate(gv.flatten(0, 1), size=(256, 256),
                                       mode="nearest").view(10, 1, 1, 256, 256).to(dev)
                    ao = anchor(xv.squeeze(1))
                    ft = ao["F_tf"].unsqueeze(1)
                    fc = ao["F_tc"].unsqueeze(1)
                    pfv = pclf.forward(ft, fc, torch.ones(9, device=dev))
                    hov = hdc(ft, pfv["fine"]["states"], pfv["coarse"]["states"])
                    dov = dec(hov["F_e"], hov["P"], hov["Q_e"])
                    sel = dov["quality"].argmax(-1).reshape(-1)
                    prob = torch.sigmoid(dov["masks"]).reshape(-1, 3, 256, 256)
                    pv = (prob[torch.arange(sel.shape[0]), sel] > 0.5).float().view(10, 1, 1, 256, 256)
                    ed_list.append(float(dice_score(pv[[0]], gs[[0]]).mean()))
                    es_list.append(float(dice_score(pv[[-1]], gs[[-1]]).mean()))
                    for i in range(10):
                        gi, pi = gs[i].reshape(1, 256, 256), pv[i].reshape(1, 256, 256)
                        if gi.sum() > 0 and pi.sum() > 0:
                            hd_list.append(float(hd95(pi, gi).mean()))
                    dr_list.append(float(centroid_drift(pv).mean()))
                    rg_list.append(float(area_roughness(pv).mean()))
                    va_list.append(float(interframe_dice_variation(pv).mean()))
            import numpy as _np
            mlflow.log_metric("train_loss", tl, step=ep)
            mlflow.log_metric("train_dice", td, step=ep)
            if W.get("n", 0) > 0:
                wn = max(W["n"], 1)
                for k in ("ce", "dice", "rec", "smooth", "flow", "iou", "boundary"):
                    mlflow.log_metric(f"loss/{k}_ep", W[k] / wn, step=ep)
            mlflow.log_metric("val_loss", vl, step=ep)
            mlflow.log_metric("val_dice", vd, step=ep)
            mlflow.log_metric("lr", opt.param_groups[0]["lr"], step=ep)
            mlflow.log_metric("val_hd95", float(_np.mean(hd_list)) if hd_list else -1, step=ep)
            mlflow.log_metric("val_drift", float(_np.mean(dr_list)), step=ep)
            mlflow.log_metric("val_rough", float(_np.mean(rg_list)), step=ep)
            mlflow.log_metric("val_ifvar", float(_np.mean(va_list)), step=ep)
            mlflow.log_metric("val_ED", float(_np.mean(ed_list)), step=ep)
            mlflow.log_metric("val_ES", float(_np.mean(es_list)), step=ep)
            if args.diag_every > 0 and (ep % args.diag_every == 0 or ep == args.epochs - 1):
                # W-event diagnose snapshot: gamma archive from one val clip
                with torch.no_grad():
                    xv0, _ = load_patient(vids[0])
                    xv0 = xv0.to(dev)
                    ao0 = anchor(xv0.squeeze(1))
                    ft0 = ao0["F_tf"].unsqueeze(1)
                    fc0 = ao0["F_tc"].unsqueeze(1)
                    pf0 = pclf.forward(ft0, fc0, torch.ones(9, device=dev))
                    ho0 = hdc(ft0, pf0["fine"]["states"], pf0["coarse"]["states"],
                              record_gamma=True)
                    glog = [g.detach().cpu() for g in ho0["gamma_log"][0].unbind(0)]
                    from diag.hdc import adjacent_ratio_cv as _arcv
                    cv = _arcv(glog)["cv"]
                mlflow.log_metric("diag/gamma_cv_val", cv, step=ep)
            print(f"ep{ep:03d} train_loss={tl:.4f} train_dice={td:.4f} val_loss={vl:.4f} val_dice={vd:.4f} "
                  f"lr={opt.param_groups[0]['lr']:.1e} t={time.time()-t0:.0f}s", flush=True)
            ckpt = {"epoch": ep, "anchor": anchor.state_dict(), "pclf": pclf.state_dict(),
                    "hdc": hdc.state_dict(), "dec": dec.state_dict(), "opt": opt.state_dict(),
                    "ds_head": ds_head.state_dict(), "frame_only": bool(args.frame_only),
                    "val_dice": vd, "code_sha": code_sha}
            torch.save(ckpt, OUT / "last.pt")
            if vd > best_val:
                best_val = vd
                torch.save(ckpt, OUT / "best.pt")
                mlflow.log_metric("best_val_dice", best_val, step=ep)
        print(f"DONE best_val_dice={best_val:.4f} ckpt={OUT}/best.pt run={run.info.run_id}", flush=True)


if __name__ == "__main__":
    main()
