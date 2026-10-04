export MLFLOW_TRACKING_URI=http://172.16.240.77:5000 DIAG_OUT=/root/DIAG_fresh/outputs/stage2full HF_HUB_OFFLINE=1
cd /root/DIAG_fresh && git pull -q && mkdir -p outputs/stage2full && PYTHONPATH=src:DIAG-code python3 DIAG-code/train_camus.py --epochs 30 --batch 2 --seed 0 --ds-lr 1e-4 --l-ce 0.05 --l-rec 1.0 --run-name s2-full-ce005-rec10-s0 > outputs/stage2full/s0.log 2>&1
