"""SVF promotion grid search (004 T1-T3 + sec.6 discipline).

Grid: ss_steps {4,6,8} x max_disp {0.03,0.05,0.08} x svf_smooth {0.001,0.01}
= 18 runs, short budget (subset patients, few epochs). Promotion (003):
flow_delta > 0 + ds no-collapse + identity-phi Dice unchanged + gamma_cv
no worse. Else DELETE SVF branch. Best point re-run once at independent
seed before promotion. All params + code_sha + split md5 + seed logged.
"""
import itertools
import subprocess
import sys
from pathlib import Path

GRID = {
    "ss_steps": [4, 6, 8],
    "max_disp": [0.03, 0.05, 0.08],
    "svf_smooth": [0.001, 0.01],
}
BUDGET_EPOCHS = 6
SUBSET = 60
SEED_MAIN = 0
SEED_RERUN = 7


def main():
    combos = list(itertools.product(GRID["ss_steps"], GRID["max_disp"], GRID["svf_smooth"]))
    print(f"grid={len(combos)} runs x {BUDGET_EPOCHS}ep subset={SUBSET}", flush=True)
    if len(combos) > 18:
        print("OVER BUDGET: stop", flush=True)
        sys.exit(2)
    for i, (ss, md, sw) in enumerate(combos):
        name = f"svf-g-{ss}-{md}-{sw}"
        cmd = (f"cd /root/DIAG && export MLFLOW_TRACKING_URI=http://172.16.240.77:5000 && "
               f"python3 DIAG-code/train_camus.py --epochs {BUDGET_EPOCHS} --subset {SUBSET} --batch 2 "
               f"--svf --ss-steps {ss} --max-disp {md} --svf-smooth {sw} --seed {SEED_MAIN} "
               f"--run-name {name} > outputs/camus_train/{name}.log 2>&1")
        print(f"[{i+1}/{len(combos)}] {name}", flush=True)
        subprocess.run(["ssh", "-o", "BatchMode=yes", "-p", "32237", "root@172.16.240.188", cmd], check=True)
    print("GRID DONE; inspect mlflow then rerun best at seed 7", flush=True)


if __name__ == "__main__":
    main()
