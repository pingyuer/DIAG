export MLFLOW_TRACKING_URI=http://172.16.240.77:5000 DIAG_OUT=/root/DIAG_fresh/outputs/stage2 HF_HUB_OFFLINE=1
cd /root/DIAG_fresh && git pull -q && mkdir -p outputs/stage2 && PYTHONPATH=src:DIAG-code python3 DIAG-code/train_camus.py --epochs 6 --subset 60 --batch 2 --seed 1 --ds-lr 1e-4 --l-ce 0.05 --l-rec 0.5 --run-name s2-ce0.05-rec0.5-s1 > outputs/stage2/ce0.05-rec0.5-s1.log 2>&1
