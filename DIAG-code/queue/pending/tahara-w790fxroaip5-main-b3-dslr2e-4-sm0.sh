export MLFLOW_TRACKING_URI=http://172.16.240.77:5000 DIAG_OUT=/root/DIAG_fresh/outputs/b3_dslr2e-4_sm0
cd /root/DIAG_fresh && git pull -q && mkdir -p outputs/b3_dslr2e-4_sm0 && PYTHONPATH=src:DIAG-code python3 DIAG-code/train_camus.py --epochs 6 --subset 60 --batch 2 --seed 0 --ds-lr 2e-4 --l-ce 0.05 --l-rec 1.0 --l-smooth 0 --run-name b3-dslr2e-4-sm0 > outputs/b3_dslr2e-4_sm0/run.log 2>&1
