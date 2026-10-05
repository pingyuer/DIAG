export MLFLOW_TRACKING_URI=http://172.16.240.77:5000 DATASETS_ROOT=/input0
mkdir -p /root/DIAG_fresh/outputs && rm -rf /root/DIAG_fresh/upstream_BanditPM && tar xzf /tmp/upstream.tgz -C /root/DIAG_fresh && cd /root/DIAG_fresh/upstream_BanditPM && bash train.sh --config-name=gdkvm_camus_fair_dense10 seed=0 mlflow.experiment_name=gdkvm-compare mlflow.run_name=gdkvm-fair-s0 mlflow.tags.first_frame_gt=false > /root/DIAG_fresh/outputs/queue-c2-gdkvm-s0.log 2>&1
echo "c2-s0 metrics:"; grep -E "test_dice|test_hd95|Dice|HD95" /root/DIAG_fresh/outputs/queue-c2-gdkvm-s0.log | tail -5
