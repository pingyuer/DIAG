export MLFLOW_TRACKING_URI=http://172.16.240.77:5000 DATASETS_ROOT=/input0
cd /root/DIAG_fresh/upstream_BanditPM && git pull -q; bash train.sh --config-name=gdkvm_camus_fair_dense10 > /root/DIAG_fresh/outputs/queue-c2-gdkvm-fair.log 2>&1
