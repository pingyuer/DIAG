export MLFLOW_TRACKING_URI=http://172.16.240.77:5000 DIAG_OUT=/root/DIAG_fresh/outputs/grid2 HF_HUB_OFFLINE=1
cd /root/DIAG_fresh && git pull -q && mkdir -p outputs/grid2 && PYTHONPATH=src:DIAG-code python3 DIAG-code/train_camus.py --epochs 30 --batch 2 --seed 1 --ds-lr 1e-4 --no-detach-region-s --run-name grid2-off-s1 > outputs/grid2/off-s1.log 2>&1
