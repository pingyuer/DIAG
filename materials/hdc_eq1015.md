# HDC Eq.10-15 验收（eval n-20260930-142824-552 ← impl n-20260930-141014-0a4, commit 57f2c86）

口径依据：`proposals/000_diag_paper_parse.md` §1.3（Eq.10 θ_t 六元组、A 仿射门控 γ>0、 B dense prompt 叠加、 C region token 门控并入）+ `proposals/001_diag_innovations.md` §3 HDC-A/B/C 清单项（禁 FiLM `(1+scale)`、禁 logits 融合、留 H4 诊断钩子）。

## 1. 结论：通过（带三条如实记录，不打回）

- γ 恒正、α∈(0,1)、近恒等启动、梯度全覆盖、对角复合恒等式、远端 CUDA——父节点断言全部属实，本地独立重跑逐项复现。
- 记录 A（未训练退化值，非缺陷）：真 clip 端到端 γ 全单元恒为 0.8808（=sigmoid(2.0)，零权初值所致），故 adjacent-ratio CV=0.0。这与论文 Fig.3c 的 ~1e-6（训练后 γ）不可比；smoke 脚本注释已声明"untrained: smoothness only, no target value"。CV 诊断链路（archive→`adjacent_ratio_cv`→`compose_gain`）已验证可跑，数值等待训练后复测。
- 记录 B（重复 run）：exp96 内 `smoke-hdc-eq1015` 有两个同 sha run（6f56fc63 / f3276ddad，同 code_sha=4187dc50…，同指标）。与 PCLF 节点同类情况一致，系 smoke 脚本重跑；验收以后者（f3276ddad）为准。
- 记录 C（实现约束）：`HDC.__init__` 有 `assert coarse == fine`（region 双头共享 Q_0 所致）。当前 anchor 双尺度同为 96 通道，不触发；若后继基座更换致通道数分叉，需先投影 coarse。本约束记于此，不在本节点解决。

## 2. 实测复核（本地 .venv torch 2.14 cpu，真 clip 10 帧，seed 0）

| 断言 | 实测 |
|---|---|
| F_e/P/Q_e 形状 == (10,1,96,128,128)/(同)/(10,1,4,96) | ✅ |
| γ 全员 >0 | ✅ min=max=0.8808（初值），mean 一致 |
| α_c=α_f≈0.12 ∈(0,1) | ✅ 0.119203（=sigmoid(-2.0)） |
| 近恒等启动：A γ≈0.88/β=0；B dense 头零初值；Q_e≈Q_0 | ✅ γ 初值 0.8808/β max=0；Q_e−Q_0 max=0.057（≈0.12·\|T\| 量级） |
| 全参数有梯度 | ✅ backward 后无 grad=None 参数 |
| 对角复合 R13==R23·R12 | ✅ maxdiff=0.0 |
| n_params=94274 | ✅（≈94K，与 anchor 一致） |
| 禁 FiLM `(1+scale)` / 无 logits 融合 | ✅ grep 仅命中注释中的架构要求声明 |
| `git diff 57f2c86 HEAD -- src/diag/hdc.py` | ✅ 空，无漂移 |
| 远端 :32237 CUDA | ✅ F_e/P/Q_e 形状对，γ 恒正 |

## 3. 设计核验（相对 001 §2.4 的冲突清单）

- A：γ=sigmoid（恒正），非 FiLM `(1+scale)`；零权+bias 2.0 近直通启动，σ'≈0.105 梯度健康。
- B：双 dense 头零初始化，P_t=P_0（默认可学习零 map）恒等启动；P_0 外部注入接口保留（预训练 segmenter 到位后接）。
- C：K=4 query 注意力，K/V 独立投影，α=sigmoid(raw=-2.0)，Q_0 trunc-normal(0.02)。
- 调制全在 decoder 之前（frame-wise over (T,B,...)，无时序混合；动力学只活在 PCLF）。
- H4 钩子：`record_gamma` archive + `compose_gain` + `adjacent_ratio_cv` 三件套齐；impl anchor 已诚实注明"纯 γ 算术可交换，有序/shuffle 诊断须经特征做"——该诊断属训练后 evaluation 范围。

## 4. mlflow run（exp96）

- run `smoke-hdc-eq1015` ×2（见记录 B），`tags.node=hdc-eq1015`，`params.num_queries=4`，`metrics.gamma_min=0.880797, alpha_c=0.119203, n_params=94274.0`——与本地重跑一致。
- code_sha 谱系：run sha=4187dc50…（HDC 落地时）→ 当前 `ca71e401…`，后继训练/推理提交的正常推进。

## 5. 与论文结论的对应关系

- 本节点是 Eq.10–15 的结构存在性验证，不产生 Table 1–2 对照数值。
- H3 验收（Table 2 单项移除 −0.78/−0.46/−0.17）需等待 decoder + 训练 + CAMUS 数据管线就绪后，在训练模型上做消融方可用；当前恒等启动状态的三分支输出不可比。
- H4 验收（CV 量级、有序/shuffle 2.13×）需训练后 γ 方可度量；当前 CV=0.0 **禁止**引用为论文对照值。
