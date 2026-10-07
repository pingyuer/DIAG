# DIAG — Dynamics-Induced Affine Gating（超声心动视频分割）

论文依据：`paper/references/35002_DIAG_Dynamics_Induced_Af.pdf`（本地备查，不进仓）。

## 两套跑法

### A. 本地（py3.11, uv）

```bash
uv sync
uv run python DIAG-code/smoke_anchor.py
```

torch 用 CPU 源（`pyproject.toml [tool.uv.sources]`），只做验证不过训练。

### B. 容器（py3.10 + torch2.6cu124 现成，不建 venv）

```bash
cd /root/DIAG
pip install mlflow  # 一次性
export MLFLOW_TRACKING_URI=http://172.16.240.77:5000
git pull
PYTHONPATH=src:DIAG-code python3 DIAG-code/train_camus.py --epochs 30 --batch 2
```

脚本内已有 `sys.path.insert(0, "src"/"DIAG-code")`，`PYTHONPATH` 双保险；
数据只读 `/input0-3`，产物落 `outputs/`（gitignored）。

## 打标口径

- 每个 mlflow run 必带 `code_sha`（=`git rev-parse --short HEAD`，过渡期 `sync.sh sha` 同值）。
- 阈值表见 `train_camus.py: W_THRESHOLDS`（O1–O10），写 run tags。
- 偏差（随机基座/占位λ/dts=ones）写 run tags，不口头约定。

## 链路

`ContentAnchor` → `PCLF` → `HDC` → `CandidateDecoder` → `DiagLoss`，
`DsHead` 供步长，`svf.py` 诊断工具（已从训练链摘除）。
详见 `proposals/000_diag_paper_parse.md`（Eq.5–17 锚点）。
