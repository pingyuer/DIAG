export MLFLOW_TRACKING_URI=http://172.16.240.77:5000 DIAG_OUT=/root/DIAG_fresh/outputs/dino_s1 HF_HUB_OFFLINE=1
cd /root/DIAG_fresh && git pull -q && mkdir -p outputs/dino_s1 && PYTHONPATH=src:DIAG-code python3 DIAG-code/train_camus.py --epochs 30 --batch 2 --anchor dinov2-small --seed 1 --ds-lr 1e-4 --run-name dino-s-full-s1 > outputs/dino_s1/run.log 2>&1
