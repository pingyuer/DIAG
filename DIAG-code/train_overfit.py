"""Single-clip overfit: proves the optimization path is alive.

Same full chain as train_camus.py (Anchor->PCLF->HDC->Decoder->DiagLoss),
one patient, CPU-or-GPU, few hundred steps. Pass = total loss falls clearly
and WTA Dice rises on the SAME clip. Values are meaningless vs the paper
(random-init backbone + placeholder lambdas); only the DIRECTION matters.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import mlflow
import torch
import torch.nn.functional as F

sys.path.insert(0, "src")
sys.path.insert(0, "DIAG-code")
from diag_metrics import dice_score
from diag.anchoring import BACKBONE_RECORD, ContentAnchor
from diag.data_camus import DEVIATIONS, load_patient
from diag.decoder import CandidateDecoder
from diag.hdc import HDC
from diag.losses import DiagLoss, DiagLossWeights
from diag.pclf import PCLF

MLFLOW_URI = "http://172.16.240.77:5000"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patient", default="patient0001")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--run-name", default="overfit-1clip")
    args = ap.parse_args()

    dev = torch.device(args.device if torch.cuda.is_available() or "cpu" in args.device else "cpu")
    if "cuda" in args.device and not torch.cuda.is_available():
        dev = torch.device("cpu")
    torch.manual_seed(0)
    print(f"device={dev} patient={args.patient} steps={args.steps}", flush=True)

    clip = load_patient(args.patient)
    x = clip["images"].to(dev)
    g = clip["masks"].to(dev)
    dts = clip["dts"].to(dev)
    lab = torch.ones(10, 1, dtype=torch.bool, device=dev)

    anchor = ContentAnchor().to(dev)
    C = anchor.out_channels
    pclf = PCLF(C).to(dev)
    hdc = HDC(C, num_queries=4).to(dev)
    dec = CandidateDecoder(feat_dim=C, num_queries=4, num_candidates=3).to(dev)
    loss_fn = DiagLoss(DiagLossWeights(), state_dim=C).to(dev)
    params = list(anchor.parameters()) + list(pclf.parameters()) + list(hdc.parameters()) \
        + list(dec.parameters()) + list(loss_fn.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-2)

    try:
        sha = Path(".code_sha").read_text().strip()
    except OSError:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True).stdout.strip()
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("diag-camus-train")
    t0 = time.time()
    with mlflow.start_run(run_name=args.run_name) as run:
        mlflow.set_tag("code_sha", sha)
        mlflow.set_tag("node", "train-overfit")
        for k, v in DEVIATIONS.items():
            mlflow.set_tag(k, v)
        mlflow.log_param("patient", args.patient)
        mlflow.log_param("steps", args.steps)
        mlflow.log_param("lr", args.lr)
        hist = []
        for it in range(args.steps):
            for m in (anchor, pclf, hdc, dec, loss_fn):
                m.train()
            opt.zero_grad()
            feats = [anchor(f) for f in x]
            f_tf = torch.stack([o["F_tf"] for o in feats])
            f_tc = torch.stack([o["F_tc"] for o in feats])
            pf = pclf.forward(f_tf, f_tc, dts)
            ho = hdc(f_tf, pf["fine"]["states"], pf["coarse"]["states"])
            do = dec(ho["F_e"], ho["P"], ho["Q_e"])
            out = loss_fn(do["masks"], do["quality"], g, lab,
                          pf["fine"]["states"], pf["fine"]["obs"], dts)
            out["total"].backward()
            opt.step()
            with torch.no_grad():
                sel = do["quality"].argmax(-1).reshape(-1)
                prob = torch.sigmoid(do["masks"]).reshape(-1, 3, 256, 256)
                best = prob[torch.arange(sel.shape[0]), sel]
                d = float(dice_score((best > 0.5).float().view(10, 1, 1, 256, 256), g).mean())
            tot = float(out["total"])
            hist.append((tot, d))
            mlflow.log_metric("loss", tot, step=it)
            mlflow.log_metric("dice", d, step=it)
            if it % 20 == 0 or it == args.steps - 1:
                print(f"it{it:04d} loss={tot:.4f} dice={d:.4f} t={time.time()-t0:.0f}s", flush=True)
        l0, d0 = hist[0]
        l1, d1 = hist[-1]
        print(f"OVERFIT l0={l0:.4f}->l1={l1:.4f} d0={d0:.4f}->d1={d1:.4f}", flush=True)
        assert l1 < l0, f"loss did not fall: {l0} -> {l1}"
        assert d1 > d0, f"dice did not rise: {d0} -> {d1}"
        ckpt = {"step": args.steps, "patient": args.patient,
                "anchor": anchor.state_dict(), "pclf": pclf.state_dict(),
                "hdc": hdc.state_dict(), "dec": dec.state_dict(),
                "loss": (l0, l1), "dice": (d0, d1), "code_sha": sha}
        out_path = Path("/root/DIAG/outputs/overfit_1clip.pt") if Path("/root/DIAG").exists \
            else Path("outputs/overfit_1clip.pt")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(ckpt, out_path)
        print(f"ckpt -> {out_path}", flush=True)
        print(f"DONE run={run.info.run_id}", flush=True)


if __name__ == "__main__":
    main()
