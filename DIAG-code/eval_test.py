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
                          interframe_dice_variation, patient_average)
from diag.anchoring import ContentAnchor
from diag.decoder import CandidateDecoder
from diag.hdc import HDC
from diag.pclf import PCLF

DATA = Path("outputs/camus_test_pull")  # scp target; fallback remote path first
REMOTE = "root@172.16.240.188:/input0/processed/camus_png256_10f"
CKPT = Path("outputs/remote-32237/camus_train/best.pt")


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
    anchor = ContentAnchor().eval()
    C = anchor.out_channels
    pclf = PCLF(C).eval()
    hdc = HDC(C, num_queries=4).eval()
    dec = CandidateDecoder(feat_dim=C, num_queries=4, num_candidates=3).eval()
    anchor.load_state_dict(ckpt["anchor"])
    pclf.load_state_dict(ckpt["pclf"])
    hdc.load_state_dict(ckpt["hdc"])
    dec.load_state_dict(ckpt["dec"])

    DATA.mkdir(parents=True, exist_ok=True)
    dts = torch.ones(9)
    per_frame, per_pid, hvals, drifts, roughs, varis = [], [], [], [], [], []
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
            pred = (best > 0.5).float().view(10, 1, 1, 256, 256)
        d = dice_score(pred, g)
        per_frame.append(d)
        per_pid += [pid] * 10
        for i in range(10):
            if g[i].sum() > 0 and pred[i].sum() > 0:
                hvals.append(float(hd95(pred[i:i+1], g[i:i+1]).mean()))
        drifts.append(float(centroid_drift(pred).mean()))
        roughs.append(float(area_roughness(pred).mean()))
        varis.append(float(interframe_dice_variation(pred).mean()))
    pf = torch.cat(per_frame)
    print(f"frame-mean dice={float(pf.mean()):.4f}", flush=True)
    print(f"patient-avg dice={float(patient_average(pf, torch.tensor([hash(p) % 10**6 for p in per_pid]))):.4f} (approx ids)", flush=True)
    import numpy as np_
    print(f"hd95 mean={np_.mean(hvals):.2f} n={len(hvals)}", flush=True)
    print(f"drift={np_.mean(drifts):.2f} rough={np_.mean(roughs):.1f} var={np_.mean(varis):.4f}", flush=True)


if __name__ == "__main__":
    main()
