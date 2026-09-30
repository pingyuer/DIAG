# 环境打通验收（eval n-20260930-134632-9c1 ← impl n-20260930-133524-465）

验收对象：三工作区环境打通（本地 + 双GPU容器），不含算法指标。Dice/HD95/时序口径以
`proposals/000_diag_paper_parse.md` §3–§5（论文表1-2定义）为准，本节点无算法产物，
口径无变更、无迁就。

## 1. 结论：通过（带一条预期内漂移说明）

- 三处 venv 可用、sync.sh 双向通、mlflow exp `diag-env-bringup`（id 96）含 code_sha=`48699828…` 的三 run——父节点断言全部属实。
- 唯一漂移：当前 `code_sha=3f9b2f96…`（HEAD `3f50423`，两次后继提交 `7574344`/`3f50423` 所致），
  远端双容器 `.code_sha` 已同步为同一值，属后继工作正常 push，非本节点失败。

## 2. 环境矩阵（2026-09-30 实测，非转述）

| 位置 | python | torch | mlflow | GPU/备注 |
|---|---|---|---|---|
| 本地 `.venv` | 3.11 | 2.14.0+cpu | 3.16.1 | CPU；`torch` 走 `pytorch-cpu` explicit index |
| 容器 `:32237` | 3.10（系统 pip） | 2.6.0+cu124 | 3.16.1 | NVIDIA A30，CUDA 可用 |
| 容器 `:31035` | 3.10（系统 pip） | 2.6.0+cu124 | 3.16.1 | NVIDIA A30，CUDA 可用 |

- 双容器 `~/.bashrc` 均含 `export MLFLOW_TRACKING_URI=http://172.16.240.77:5000`。
- `pyproject.toml`：`mlflow>=2, numpy, Pillow, torch` + `tool.uv.sources/index pytorch-cpu`，与断言一致。
- `.gitignore`：`mlruns/ outputs/ checkpoints/ *.ckpt *.pt *.pth .code_sha` 均覆盖；
  `.code_sha` 有意不进 git（`git show ee08d15:.code_sha` 为空符合预期）。

## 3. code_sha 谱系

| 时刻 | sha（前缀） | 说明 |
|---|---|---|
| `ee08d15`（父节点提交） | `48699828…` | 父节点三 run 所打标；本地 `outputs/remote-*/smoke_gpu.json` 仍保留该值（历史产物） |
| 当前 HEAD `3f50423` | `3f9b2f96…` | 本地 `.code_sha` == 双容器 `/root/DIAG/.code_sha`（均已验证），后继 Eq.5 工作正常 push |

## 4. mlflow exp96（`diag-env-bringup`，共 7 runs，按时间升序）

| # | run 名 | host | code_sha 前缀 | 关键指标/参数 |
|---|---|---|---|---|
| 0 | smoke-local-cpu | local-cpu | `35cd129e`（更早基线） | dummy_interframe_dice=0.960768 |
| 1–2 | smoke-remote-gpu ×2 | remote-gpu | `35cd129e` | forward_ok=1.0，torch=2.6.0+cu124，gpu=NVIDIA A30 |
| **3** | **smoke-local-cpu** | local-cpu | **`48699828` ✅** | dummy_interframe_dice=0.960768（与 run0 一致，样本确定性） |
| **4–5** | **smoke-remote-gpu ×2** | remote-gpu | **`48699828` ✅** | forward_ok=1.0，torch=2.6.0+cu124 |
| 6 | smoke-anchor-eq5 | local-cpu | `3f9b2f96`（当前） | 后继 Eq.5 节点产物，非本节点范围 |

父节点“三个 run（local 1 + remote 2，均带 code_sha）”即 #3–#5，逐 run 核对一致。

## 5. sync.sh 双向（只读核验，未重跑污染 mlflow）

- `DIAG-code/sync.sh` 自 `ee08d15` 起未改动（`git show ee08d15:DIAG-code/sync.sh | diff -` 空）。
- push 面：本地 `.code_sha` == 双容器远端值 → push 链路有效（含后继提交的同步）。
- pull 面：`outputs/remote-32237/smoke_gpu.json`、`outputs/remote-31035/smoke_gpu.json`
  均存在且 `gpu=NVIDIA A30, out=[1,4,256,256]`；远端 `/root/DIAG/outputs` 当前仅 `smoke_gpu.json`，
  与本地 pull 产物一致，无丢失、无多余。
- 本地 `outputs/smoke_sample/` 10 帧 PNG（patient0001 clip）存在；`smoke_local.py` 断言
  `(10,256,256)` + 帧间阈值 Dice 逻辑与 run 指标自洽（dummy 口径，仅占位、非论文 Dice）。

## 6. 口径声明（与论文结论的对应关系）

- 本节点是 H1–H4 复现的前置环境 prerequisite，不产生论文表 1-2 对照数值；
  后续 evaluation 节点须以 `proposals/000_diag_paper_parse.md` §3（表1：四数据集 Dice/HD95/mDice、
  checkpoint 选择、prompted 块仅参考）与 §4（表2：CAMUS ED/ES/Mean、全配置 .9427/.9218/.9317、
  单分量移除 Δ）为验收依据，seed/采样/HD95 实现缺失项见该文档 §5–§6。
- 警示：`dummy_interframe_dice`（阈值+帧间平均）与论文 Dice（标注帧、surface-based HD95 配对）
  口径完全不同，**禁止**将 0.960768 引用为任何论文对照值。
