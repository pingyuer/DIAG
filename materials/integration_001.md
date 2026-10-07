# 001 清单集成验收（eval n-20260930-144658-968 ← impl n-20260930-140939-e22）

口径依据：`proposals/001_diag_innovations.md` §3 核对清单（Anchoring / PCLF / HDC-A/B/C / 训练推理 / 指标诊断六项）+ 各子节点验收文件（`materials/anchoring_eq5.md, pclf_eq69.md, hdc_eq1015.md, train_eq1617.md, metrics_diag.md`）。本文件只做集成级复核（chain 完整、冲突件零实现、mlflow 全覆盖），不重复子节点数值。

## 1. 结论：通过

- 四子 chain（PCLF / HDC / 训练推理 / 指标诊断，全在子 commit a88e0c6 / 57f2c86 / e33213f / 0d9a9cc）完整，父节点无新 commit——`git log` 确认父审计后仅 evaluation 提交，无代码漂移。
- 冲突件零实现：独立 grep 确认（见 §2）。
- mlflow exp96 当前 17 runs，code_sha 覆盖 17/17 零缺失（父 anchor 称 16/16 系其审计时刻计数，现新增 1 个 train 重跑，见记录 A）。
- 本地全链路与远端 CUDA e2e 独立重跑通过；upstream 零修改。

## 2. 冲突件审计（独立 grep，2026-09-30）

| 禁止件 | 结果 |
|---|---|
| `grid_sample` / warp / `DPFRFlowHead` / `DPFRResidualFusion` / `ResidualFusion` | ✅ 零命中（src/diag + DIAG-code） |
| `DPFRDualPromptEncoder` / dpfr 复用 | ✅ 唯一命中为 `pclf.py:13` 注释中的禁用声明（"Do NOT reuse…"），无 import、无调用 |
| `use_gt` / `gt_prob` / teacher forcing | ✅ 唯一命中为 `smoke_train.py:33` 断言扫描的 token 表，无实现 |
| FiLM `(1+scale)` | ✅ 实现仅存在于 vendored `unext_backbone.py:198`（与 upstream 逐行一致）；`ContentAnchor` 只调 `backbone.encode()["low"]`（`anchoring.py:70`），全仓零 `.decode(` 调用点（除 backbone 自身定义）；且 `_apply_modulation` 在 `modulation=None` 时直接返回（early-return），即使误调也是 no-op。FiLM 路径不可达 |
| logits 级融合 | ✅ `logits` 命中均为"候选/quality logits"中性术语与架构要求注释（train 验收已逐条定位），无融合实现 |

`upstream_BanditPM/`：untracked 只读对照，`git status` 无 tracked 修改（仅 `??` 未跟踪项）。

## 3. 全链路独立重跑

- 本地（torch 2.14 cpu，T=4/B=1 随机输入）：Anchor→PCLF→HDC→Decoder→Loss 全通，masks=(4,1,3,256,256)，total=1.62，可反传、无 grad=None 参数。数值与父 anchor 的 total=1.89 不同属正常（输入不同：随机 vs 真 clip+伪 GT），结构一致即通过。
- 插曲即证据：重跑中两次误调（5D 直喂 anchor；模型放 CPU 而输入放 CUDA）均被正确拒绝——BCHW 守卫 `ValueError` 与 torch device 一致性报错，属守卫生效，非代码缺陷。
- 远端 :32237 CUDA（模型 `.to(cuda:0)` 后）：e2e masks=(4,1,3,256,256) ✅。
- 约束记录：`HDC.__init__` 仍有 `assert coarse == fine`（hdc 验收记录 C）；`CandidateDecoder` 默认 `feat_dim=96/num_queries=4` 须与 anchor/HDC 对齐，远端 e2e 须显式传参（父 anchor 已注明 dim mismatch 调试史）。

## 4. mlflow exp96（17 runs，code_sha 17/17）

| # | run | code_sha 前缀 | 备注 |
|---|---|---|---|
| 0–2 | smoke-local-cpu / remote-gpu ×2 | 35cd129e | 更早基线 |
| 3–5 | smoke-local-cpu / remote-gpu ×2 | 48699828 | 环境打通验收 |
| 6 | smoke-anchor-eq5 | 3f9b2f96 | Anchoring |
| 7–8 | smoke-pclf-eq69 ×2 | 4e2f6a9f | PCLF（重跑，见 pclf 验收记录 B） |
| 9–10 | smoke-hdc-eq1015 ×2 | 4187dc50 | HDC（重跑，见 hdc 验收记录 B） |
| 11–12 | smoke-train-eq1617 ×2 | ca71e401 | 训练（重跑，见 train 验收记录 B） |
| 13–14 | smoke-metrics-camus3 ×2 | 0402cd2c | 旧版 R2=0.0 bug（见 metrics 验收记录 A） |
| 15 | smoke-metrics-camus3 | 767b5ad3 | ✅ 验收有效 run（R2=0.918） |
| 16 | smoke-train-eq1617 | 767b5ad3 | 记录 A：父审计后新增的 train 重跑，同当前 HEAD |

- 记录 A：run #16（`smoke-train-eq1617` @ 767b5ad3）是父节点审计时不存在的新 run，属后继工作正常重跑；code_sha 即当前 HEAD，无漂移。
- 重复 run 模式（同名同 sha 双 run）贯穿 PCLF/HDC/训练/指标四节点，系各 smoke 脚本调试重跑所致，无覆盖冲突；各节点验收文件已分别指定有效 run。

## 5. 清单六项对照（001 §3）

| 清单项 | 状态 | 验收文件 |
|---|---|---|
| Anchoring：基座选型+权重来源记录、F_tf/F_tc=Down、UNeXt 记偏差 | ✅ | anchoring_eq5.md |
| PCLF：新建、O^q/f^q/Euler dt/S1=O1/K 门、禁复用 DPFRDualPromptEncoder | ✅ | pclf_eq69.md |
| HDC-A：γ=σ 恒正、禁 FiLM、H4 钩子 | ✅ | hdc_eq1015.md |
| HDC-B/C：dense 叠 P_0、region K-query α∈(0,1) 并 Q_0；flow warp 与 logits 融合零实现 | ✅ | hdc_eq1015.md（+本文件 §2） |
| 训练/推理：J 候选+质量选择、Eq.17 六项（λ 占位）、无 use_gt、新模块默认无 GT 路径 | ✅ | train_eq1617.md |
| 指标/诊断：Dice/HD95/时序三件套/patient 平均 + R2/干预/CV/复合/采样压力 | ✅ | metrics_diag.md |

## 6. 与论文结论的对应关系

- 本节点确认 Eq.5–17 + 指标/诊断在**结构层面闭环**（untrained 全链路可跑、可反传、可度量），不产生 Table 1–2 对照数值。
- H1–H4 实证仍缺四项前提（λ 真值、预训练基座、真实 dt 数据源、CAMUS 训练），各子验收文件已分别记录；下一步应为训练 proposal，而非更多结构节点。
