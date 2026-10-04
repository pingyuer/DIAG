export MLFLOW_TRACKING_URI=http://172.16.240.77:5000 DIAG_OUT=/root/DIAG_fresh/outputs/dino_fo HF_HUB_OFFLINE=1
cd /root/DIAG_fresh && git pull -q && mkdir -p outputs/dino_fo && PYTHONPATH=src:DIAG-code python3 DIAG-code/train_camus.py --epochs 30 --batch 2 --anchor dinov2-small --frame-only --seed 0 --ds-lr 0 --run-name dino-fo-full > outputs/dino_fo/run.log 2>&1
