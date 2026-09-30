"""容器单卡 smoke：CUDA forward + 读 processed 真实帧 + mlflow run 打标 code_sha。"""
import sys
from pathlib import Path

import mlflow
import torch
from PIL import Image

MLFLOW_URI = "http://172.16.240.77:5000"
FRAME = "/input0/processed/camus_png256_10f/img/patient0001/0000.png"


def main():
    assert torch.cuda.is_available(), "CUDA 不可用"
    dev = torch.device("cuda:0")
    im = Image.open(FRAME)
    x = torch.randn(1, 1, 256, 256, device=dev)
    conv = torch.nn.Conv2d(1, 4, 3, padding=1).to(dev)
    with torch.no_grad():
        y = conv(x)
    assert y.shape == (1, 4, 256, 256), y.shape
    print(f"gpu={torch.cuda.get_device_name(0)} frame={im.size} out={tuple(y.shape)}", flush=True)

    code_sha = Path("/root/DIAG/.code_sha").read_text().strip()
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("diag-env-bringup")
    with mlflow.start_run(run_name="smoke-remote-gpu"):
        mlflow.set_tag("code_sha", code_sha)
        mlflow.set_tag("host", "remote-gpu")
        mlflow.log_param("torch", torch.__version__)
        mlflow.log_param("gpu", torch.cuda.get_device_name(0))
        mlflow.log_metric("forward_ok", 1.0)
    out = Path("/root/DIAG/outputs")
    out.mkdir(parents=True, exist_ok=True)
    (out / "smoke_gpu.json").write_text(
        f'{{"gpu": "{torch.cuda.get_device_name(0)}", "out": [1, 4, 256, 256], "code_sha": "{code_sha}"}}')
    print(f"mlflow run ok code_sha={code_sha}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
