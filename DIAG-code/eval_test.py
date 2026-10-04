"""Test-set eval on CAMUS 50 patients with trained ckpt (local, CPU ok).

Loads pulled ckpt (outputs/remote-32237/camus_train/best.pt), runs full chain
on test split, reports frame-mean Dice + patient-averaged Dice + HD95 (non-empty)
+ temporal trio. No GT in forward (prompt-free, same as training).
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, "src")
sys.path.insert(0, "DIAG-code")
from diag_metrics import (area_roughness, centroid_drift, dice_score, hd95,
                          hd95_mirror, interframe_dice_variation,
                          patient_average, postprocess_binary_mask)
from diag.anchoring import ContentAnchor
from diag.dino_anchor import FrozenDinoAnchor
from diag.decoder import CandidateDecoder
from diag.hdc import HDC
from diag.pclf import PCLF

DATA = Path("outputs/camus_test_pull")  # scp target; fallback remote path first
REMOTE = "root@172.16.240.188:/input0/processed/camus_png256_10f"
import os
CKPT = Path(os.environ.get("DIAG_CKPT", "outputs/remote-31035/camus_train/dsfull_best.pt"))


def main():
    import subprocess
    torch.manual_seed(0)
    # read split from pulled or remote
    split = None
    for cand in (DATA / "camus_public_datasplit_20250706.json",
                 Path("/tmp/camus_split.json")):
        if cand.exists():
            split = json.loads(cand.read_text())
            break
    if split is None:
        subprocess.run(f"scp -o BatchMode=yes -P 32237 {REMOTE}/camus_public_datasplit_20250706.json /tmp/camus_split.json",
                       shell=True, check=True)
        split = json.loads(Path("/tmp/camus_split.json").read_text())
    test_ids = split["test_data"]
    print(f"test_n={len(test_ids)} ckpt={CKPT} exists={CKPT.exists()}", flush=True)

    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    print(f"ckpt epoch={ckpt['epoch']} best_val={ckpt['val_dice']:.4f} sha={ckpt.get('code_sha','?')[:8]}", flush=True)
    import argparse as _ap
    _parser = _ap.ArgumentParser()
    _parser.add_argument("--anchor", default="unext")
    _eargs, _ = _parser.parse_known_args()
    _anchor_name = _eargs.anchor
    if _anchor_name == "unext":
        anchor = ContentAnchor().eval()
    else:
        anchor = FrozenDinoAnchor(_anchor_name, out_channels=96).eval()
    C = anchor.out_channels
    pclf = PCLF(C).eval()
    hdc = HDC(C, num_queries=4).eval()
    dec = CandidateDecoder(feat_dim=C, num_queries=4, num_candidates=3).eval()
    anchor.load_state_dict(ckpt["anchor"])
    pclf.load_state_dict(ckpt["pclf"])
    hdc.load_state_dict(ckpt["hdc"])
    dec.load_state_dict(ckpt["dec"])

    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true", help="threshold sweep 0.30-0.75 on probs")
    ap.add_argument("--postprocess", action="store_true", help="upstream postprocess on masks")
    ap.add_argument("--anchor", default="unext")
    ap.add_argument("--hd95-mirror", action="store_true", dest="mirror",
                    help="report MONAI-mirror HD95 alongside cdist")
    a = ap.parse_args()

    DATA.mkdir(parents=True, exist_ok=True)
    dts = torch.ones(9)
    per_frame, per_pid, hvals, hvals_m, drifts, roughs, varis = [], [], [], [], [], [], []
    all_probs, all_gts = [], []
    for pid in test_ids:
        d = DATA / pid
        if not (d / "img").exists():
            subprocess.run(f"scp -o BatchMode=yes -q -P 32237 -r {REMOTE}/img/{pid} {REMOTE}/gt_lv/{pid} {d}/ 2>/dev/null; mkdir -p {d}/img {d}/gt_lv; scp -o BatchMode=yes -q -P 32237 {REMOTE}/img/{pid}/*.png {d}/img/; scp -o BatchMode=yes -q -P 32237 {REMOTE}/gt_lv/{pid}/*.png {d}/gt_lv/",
                           shell=True)
        imgs = sorted((d / "img").glob("*.png"))[:10]
        gts = sorted((d / "gt_lv").glob("*.png"))[:10]
        if len(imgs) < 10:
            print(f"skip {pid}: {len(imgs)} frames", flush=True)
            continue
        x = torch.stack([torch.from_numpy(np.array(Image.open(f))).float().div(255) for f in imgs]).unsqueeze(1).unsqueeze(1)
        g = torch.stack([torch.from_numpy(np.array(Image.open(f))).float() for f in gts]).unsqueeze(1).unsqueeze(1)
        g = (g > 0.5).float()
        with torch.no_grad():
            feats = [anchor(f) for f in x]
            f_tf = torch.stack([o["F_tf"] for o in feats])
            f_tc = torch.stack([o["F_tc"] for o in feats])
            pf = pclf.forward(f_tf, f_tc, dts)
            ho = hdc(f_tf, pf["fine"]["states"], pf["coarse"]["states"])
            do = dec(ho["F_e"], ho["P"], ho["Q_e"])
            sel = do["quality"].argmax(-1).reshape(-1)
            prob = torch.sigmoid(do["masks"]).reshape(-1, 3, 256, 256)
            best = prob[torch.arange(sel.shape[0]), sel]
            all_probs.append(best.detach().cpu())
            all_gts.append(g.cpu())
            pred = (best > 0.5).float().view(10, 1, 1, 256, 256)
            if a.postprocess:
                pred = postprocess_binary_mask(pred)
        d = dice_score(pred, g)
        per_frame.append(d)
        per_pid += [pid] * 10
        for i in range(10):
            if g[i].sum() > 0 and pred[i].sum() > 0:
                hvals.append(float(hd95(pred[i:i+1], g[i:i+1]).mean()))
                if a.mirror:
                    hvals_m.append(float(hd95_mirror(pred[i:i+1], g[i:i+1]).mean()))
        drifts.append(float(centroid_drift(pred).mean()))
        roughs.append(float(area_roughness(pred).mean()))
        varis.append(float(interframe_dice_variation(pred).mean()))
    pf = torch.cat(per_frame)
    # R1: true patient ids (arange repeat, NOT hash): per-patient mean then mean.
    # R2(a): first/last-frame naming (NOT ED/ES; ED/ES needs official metadata).
    n_pat = len(per_frame)
    pid_idx = torch.arange(n_pat).repeat_interleave(10)
    per_pat_dice = torch.stack([per_frame[i].mean() for i in range(n_pat)])
    print(f"frame-mean dice={float(pf.mean()):.4f}", flush=True)
    print(f"patient-avg dice={float(per_pat_dice.mean()):.4f}", flush=True)
    print(f"first-frame dice={float(torch.stack([per_frame[i][0].mean() for i in range(n_pat)]).mean()):.4f}", flush=True)
    print(f"last-frame dice={float(torch.stack([per_frame[i][-1].mean() for i in range(n_pat)]).mean()):.4f}", flush=True)
    # 008-2 hard slices: ES-proxy (min-area frame) / small-cavity / boundary band.
    # Full-mean saturates at 0.91; slices separate the pack.
    es_d, small_d, band_d, es_n, small_n = [], [], [], 0, 0
    for i in range(n_pat):
        pass  # filled below (needs probs+gts per patient; see all_probs/all_gts)
    P_all = torch.cat(all_probs) if all_probs else None
    G_all = torch.cat(all_gts).squeeze(1).squeeze(1) if all_gts else None
    if P_all is not None:
        areas = G_all.reshape(-1, 10, 256, 256).sum(dim=(2, 3))  # (P,10)
        es_idx = areas.argmin(dim=1)
        for pi in range(n_pat):
            ei = int(es_idx[pi])
            es_d.append(float(dice_score((P_all[pi * 10 + ei : pi * 10 + ei + 1] > 0.5).float(), G_all[pi * 10 + ei : pi * 10 + ei + 1]).mean()))
            es_n += 1
            a = float(areas[pi].mean())
            if a < 4100:  # small cavity (bottom quartile, ~4060px on 256px)
                small_d.append(float(dice_score((P_all[pi * 10 : (pi + 1) * 10] > 0.5).float(), G_all[pi * 10 : (pi + 1) * 10]).mean()))
                small_n += 1
        # boundary band: dilate GT edge x2, score only inside band
        import torch.nn.functional as _F
        Gb = G_all.reshape(-1, 1, 256, 256)
        dil = _F.max_pool2d(Gb, 5, stride=1, padding=2)
        ero = 1 - _F.max_pool2d(1 - Gb, 5, stride=1, padding=2)
        band = ((dil - ero) > 0).float()
        Pb = (P_all.reshape(-1, 1, 256, 256) > 0.5).float()
        inter = (Pb * band * Gb).sum()
        union = ((Pb * band).sum() + (band * Gb).sum()).clamp(min=1e-6)
        band_dice = float(2 * inter / union)
        print(f"ES-proxy dice={float(sum(es_d) / max(es_n, 1)):.4f} n={es_n}", flush=True)
        print(f"small-cavity dice={float(sum(small_d) / max(small_n, 1)):.4f} n={small_n}", flush=True)
        print(f"boundary-band dice={band_dice:.4f}", flush=True)
    # R3: split md5 + seed tags for rerun lock
    import hashlib
    split_md5 = hashlib.md5(Path("/tmp/camus_split.json").read_bytes()).hexdigest()[:8] \
        if Path("/tmp/camus_split.json").exists() else "remote-unfetched"
    print(f"split_md5={split_md5} seed=0", flush=True)
    import numpy as np_
    print(f"hd95 mean={np_.mean(hvals):.2f} n={len(hvals)}", flush=True)
    if a.mirror:
        print(f"hd95_mirror mean={np_.mean(hvals_m):.2f} n={len(hvals_m)}", flush=True)
    print(f"drift={np_.mean(drifts):.2f} rough={np_.mean(roughs):.1f} var={np_.mean(varis):.4f}", flush=True)
    if a.sweep:
        import csv
        # OOM fix: all_probs (500,256,256) vs all_gts (500,1,1,256,256)
        # broadcast to (500,1,500,256,256) = 61GB. Squeeze G first.
        P = torch.cat(all_probs)  # (500,256,256)
        G = torch.cat(all_gts).squeeze(1).squeeze(1)  # (500,256,256)
        assert P.shape == G.shape, (P.shape, G.shape)
        rows = []
        for th in (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75):
            dth = float(dice_score((P > th).float(), G).mean())
            rows.append((th, dth))
            print(f"th={th:.2f} dice={dth:.4f}", flush=True)
        best = max(rows, key=lambda r: r[1])
        print(f"best_th={best[0]:.2f} dice={best[1]:.4f} (default report stays 0.5)", flush=True)
        with open("outputs/threshold_sweep.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["threshold", "dice"])
            w.writerows(rows)


if __name__ == "__main__":
    main()
