export MLFLOW_TRACKING_URI=http://172.16.240.77:5000 DIAG_OUT=/root/DIAG_fresh/outputs/vfabc HF_HUB_OFFLINE=1
cd /root/DIAG_fresh && PYTHONPATH=src:DIAG-code python3 DIAG-code/train_camus.py --epochs 6 --subset 60 --batch 2 --seed 0 --ds-lr 1e-4 --vf-deep --run-name vf-a-deep > outputs/vfabc/a.log 2>&1
