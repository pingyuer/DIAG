export MLFLOW_TRACKING_URI=http://172.16.240.77:5000 DATASETS_ROOT=/input0
cd /root/DIAG_fresh && rm -rf upstream_BanditPM && tar xzf /tmp/upstream.tgz && cd upstream_BanditPM && bash train.sh --config-name=gdkvm_camus_fair_dense10 seed=1 mlflow.experiment_name=gdkvm-compare mlflow.run_name=gdkvm-fair-s1 mlflow.tags.first_frame_gt=false > /root/DIAG_fresh/outputs/queue-c3-gdkvm-s1.log 2>&1
echo "c3-s1 metrics:"; grep -E "test_dice|test_hd95|Dice|HD95" /root/DIAG_fresh/outputs/queue-c3-gdkvm-s1.log | tail -5
