"""Lambda coarse grid (004 sec.6): magnitudes first, then refine.

Stage 1 (this driver): one-factor-at-a-time from placeholder center
(ce=1/dice=1/rec=0.1/smooth=0.01/flow=0.1/iou=0.5), each swept {0.01,0.1,1.0}
keeping others at center: 6 dims x 3 = 18 runs + 1 center rerun = 19 runs.
Short budget (subset patients, few epochs). Compare val_dice/val_hd95 +
loss/*_ep splits. Best dims -> stage-2 refine (separate node).
Budget guard: >19 runs stops. Best point seed-rerun before any citation.
"""
import subprocess
import sys

CENTER = {"ce": 1.0, "dice": 1.0, "rec": 0.1, "smooth": 0.01, "flow": 0.1, "iou": 0.5}
LEVELS = [0.01, 0.1, 1.0]
BUDGET_EPOCHS = 6
SUBSET = 60
SEED = 0
FLAG = {"ce": "--l-ce", "dice": "--l-dice", "rec": "--l-rec",
        "smooth": "--l-smooth", "flow": "--l-flow", "iou": "--l-iou"}


def main():
    combos = [("center", dict(CENTER))]
    for dim in CENTER:
        for lv in LEVELS:
            if lv == CENTER[dim]:
                continue
            w = dict(CENTER)
            w[dim] = lv
            combos.append((f"L-{dim}-{lv}", w))
    print(f"lambda grid={len(combos)} runs x {BUDGET_EPOCHS}ep subset={SUBSET}", flush=True)
    assert len(combos) <= 19, "over budget"
    for i, (name, w) in enumerate(combos):
        flags = " ".join(f"{FLAG[d]} {v}" for d, v in w.items())
        cmd = (f"cd /root/DIAG && export MLFLOW_TRACKING_URI=http://172.16.240.77:5000 && "
               f"python3 DIAG-code/train_camus.py --epochs {BUDGET_EPOCHS} --subset {SUBSET} --batch 2 "
               f"{flags} --seed {SEED} --run-name {name} > outputs/camus_train/{name}.log 2>&1")
        print(f"[{i+1}/{len(combos)}] {name} {w}", flush=True)
        subprocess.run(["ssh", "-o", "BatchMode=yes", "-p", "31035", "root@172.16.240.188", cmd], check=True)
    print("LAMBDA GRID DONE", flush=True)


if __name__ == "__main__":
    main()
