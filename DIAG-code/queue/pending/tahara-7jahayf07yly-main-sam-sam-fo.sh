export MLFLOW_TRACKING_URI=http://172.16.240.77:5000 DIAG_OUT=/root/DIAG_fresh/outputs/sam_fo HF_HUB_OFFLINE=1
cd /root/DIAG_fresh && git pull -q && mkdir -p outputs/sam_fo && PYTHONPATH=src:DIAG-code python3 DIAG-code/train_camus.py --epochs 6 --subset 60 --batch 2 --anchor sam --frame-only --seed 0 --ds-lr 0 --run-name sam-sam-frameonly-6ep > outputs/sam_fo/sam.log 2>&1
