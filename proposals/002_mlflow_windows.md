# mlflow 中间记录窗口设计：观察–调整闭环（CAMUS 训练）

依据：`proposals/000_diag_paper_parse.md`（§1–§5 验收锚点）、`proposals/001_diag_innovations.md`（§3 checklist）、`DIAG-code/train_camus.py` 现状（epoch 级 train_loss/train_dice/val_loss/val_dice/lr/best_val_dice + code_sha/node/backbone/lambda/dts tags）、`src/diag/losses.py`（输出 total/ce/dice/rec/smooth/flow/iou/best）、`src/diag/pclf.py`（gates/K）、`src/diag/hdc.py`（gamma/beta/alpha_c/alpha_f/gamma_log）、`DIAG-code/diag_metrics.py`（Dice/HD95/centroid drift/area roughness/inter-frame Dice variation）、`DIAG-code/diag_diagnose.py`（state intervention/gamma CV/ordered-vs-shuffled composition）。
只设计、不实现。幂等核对：`proposals/` 下无 002 文档；`materials/` 仅实现沉淀（env/metrics/pclf/hdc/train/integration），无窗口设计文档。

## 1. 记录窗口（三层）

### W-step（每 N step，默认 N=50，单 clip 过拟合阶段 N=10）
| 指标 | 来源 | 用途 |
|---|---|---|
| `loss/{total,ce,dice,rec,smooth,flow,iou}` | `DiagLoss.forward` 已返回六项拆分（现 train 只记 total，进窗口必须拆开记） | Eq.17 训练健康度；某项塌/爆直接定位 |
| `lr`、`grad_norm`（clip 前后各一） | opt.param_groups / clip_grad_norm 返回值（需新加） | 优化稳定性 |
| `K_mean`、`K_hist`（PCLF 校正门均值+直方图） | `pf[fine/coarse].gates`（T-1,B,C,H,W 现有输出） | Eq.8–9 平衡：历史 vs 当前证据 |
| `gamma_mean`、`gamma_cv_step`、`gamma_sat_ratio`（γ>0.95 占比） | `hdc` 输出 gamma（现有） | Eq.11 恒正门控健康度；H4 前兆 |
| `best_idx_hist`（WTA winner 直方图，J=3） | `loss out[best]`（现有） | Eq.16 候选分化：是否某候选垄断/某候选从不赢 |
| `alpha_c`、`alpha_f` | hdc 输出（现有标量） | Eq.15 region 门控是否塌边 |

### W-epoch（每 epoch，train/val 各一）
| 指标 | 验收挂钩 |
|---|---|
| `train/val_dice`（argmax-quality 选择口径，现状已有） | Table 1 CAMUS .9360；首要 early-stop 信号（现状 ReduceLROnPlateau 已用 val Dice） |
| `HD95`（surface-based 统一实现，`diag_metrics.py` 现有） | Table 1 CAMUS 5.43；与 Dice 双轨（论文强调 overlap≠boundary） |
| `centroid_drift`、`area_roughness`、`interframe_dice_var`（val clip 级均值+95%CI 现状缺 CI，加） | Fig.4：frame-wise/DIAG 比值 7.9×/2.3×/12.5×/3.8×；时序相干主验收 |
| `ED/ES/Mean`（端点 Dice + 十帧均值，CAMUS dense 标签可算） | Table 2 全配置 .9427/.9218/.9317；消融对照口径 |
| `loss/*_ep` 六项 epoch 均值 | 与 W-step 对齐，看整 epoch 趋势 |

### W-event（事件触发即记）
| 事件 | 记录内容 |
|---|---|
| ckpt（best + 每 K epoch 定期） | `val_dice`、`code_sha`、config 快照（args 全量）、backbone/lambda/dts tags（现状已有 tags，config 快照缺） |
| 偏差记录 | 沿用 001 §3 口径：UNeXt 随机初始化、dts=ones、λ 占位；事件发生时 `mlflow.set_tag("deviation", ...)` |
| code_sha+config 快照 | 每次 run 起点必记（现状 code_sha 已有；args 全量 log_param 缺，需补） |
| 诊断快照（每 K epoch，如 K=5） | `gamma_cv`（H4：目标相邻比率 CV~1e−6 量级）、`state_intervention_drop`（H2：shuffle→.9127/噪声→.8676 量级）、ordered-vs-shuffled 复合误差（H4：.0063 vs .0134, 2.13×）、采样压力保留率（H2：5 帧 ≥98.9% 等，`diag_diagnose.py` 现有函数） |

## 2. 观察–调整闭环判断表（imp 可直接执行）

每个窗口超阈值 → 动作。阈值为 CAMUS 首训启发值，写死进 run tag 以便回溯。

| # | 观察（窗口+阈值） | 含义 | 动作 |
|---|---|---|---|
| O1 | W-step：`gamma_sat_ratio > 0.3` 持续 200 step | γ 饱和，门控初始化/输入尺度不当 | 查 `AffineGate.gamma_bias_init`（现 2.0）与 gate 输入范数；先降 bias_init，不改架构 |
| O2 | W-step：`K_mean → 0`（<0.05）持续 200 step | 观测被忽略，退化为纯历史外推 | 查 Δt 供给（现 ones 偏差）与 `ObsEncoder` 输出范数；确认 `O(x_t)` 方差非零 |
| O3 | W-step：`K_mean → 1`（>0.95）持续 200 step | 历史被忽略，退化为帧级模型 | 查 `VectorField` 输出范数（是否坍缩）与 smooth 权重是否过大 |
| O4 | W-step：`best_idx_hist` 某候选占比 >0.95 持续 500 step | WTA 垄断，J=3 候选未分化 | 查 quality head 初始化（现 0-init）与 `l_iou` 量级；考虑 STF 阶段影响，后续提议再定 |
| O5 | W-epoch：`val_dice` 停滞（patience 内提升 <0.001） | 容量/优化/λ 问题 | 先调 λ（`DiagLossWeights` 占位，supplement 待补前只在相邻量级试）；再查 lr schedule（现状 plateau patience 5） |
| O6 | W-epoch：`train_dice ↑` 但 `val_dice ↓` 超 3 epoch | 过拟合（CAMUS 400 train patients 小数据） | 加 weight_decay/早停；记偏差，不动架构 |
| O7 | W-epoch：Dice 达标但 `HD95` 高于 5.43 超 10% | overlap≠boundary（论文 §4 明示） | 查 decoder refinement 与 region token（α 是否塌边）；不只追 Dice |
| O8 | W-event 诊断：`gamma_cv` 高出 1e−6 量级 ≥100× | H4 正则性缺失 | 查 γ 头是否被 FiLM 式改动污染；确认 `σ(·)` 恒正路径唯一 |
| O9 | W-event 诊断：`state_intervention_drop` 接近 0 | 下游不用状态（H2 失效） | 查 PCLF→HDC→decoder 通路是否被短路（如融合权重把时序分支关掉）；修通路，不加新分支 |
| O10 | W-event 诊断：采样压力保留率低于 Fig.5（5 帧 <98.9% 等） | Δt/因果性问题 | 查 dts 供给（现 ones 偏差首要嫌疑）；确认推理无未来帧泄漏 |

动作优先级：O2/O3/O9（通路级）> O1/O8（门控级）> O5/O6（优化级）> O7（边界精修）。
任何动作若改架构，必须先记偏差 tag 再改。

## 3. 与 001 验收挂钩（evaluator 按窗口复核）

| H | 验收数（000 §6/001 §1） | 对应窗口 |
|---|---|---|
| H1 | frame-only 缺口 ~2.63pt；CardiacUDA prompt-free 最优 | W-epoch ED/ES/Mean + 消融对照行（后续 frame-only run 必须同窗口口径） |
| H2 | Fig.5 保留率 + Fig.4 相干增益（drift −41.7%/−32.4% 等）+ R²=.983 + 干预 drop | W-epoch 时序三件套 + W-event 诊断快照 |
| H3 | 三项移除 −0.78/−0.46/−0.17 有序互补 | W-epoch ED/ES/Mean（后续三行消融 run 同窗口） |
| H4 | γ CV~1e−6 + 有序/shuffle 2.13× | W-step `gamma_cv_step` 前兆 + W-event 复合误差终验 |

## 4. 给 implementation 的落地清单（最小改动面）

1. `train_camus.py`：step 循环内每 N step 记 W-step 六项损失拆分（`out` 已有 ce/dice/rec/smooth/flow/iou，直接 `log_metric`）+ `K_mean`（`pf` gates 均值）+ `gamma_mean/sat_ratio`（`ho` gamma）+ `best` 直方图累积；epoch 末记 W-epoch（含 HD95/时序三件套/ED-ES-Mean，现缺的由 `diag_metrics.py` 取）；诊断快照每 K epoch 调 `diag_diagnose.py`。
2. 参数快照：`args` 全量 `log_param` + 现有 tags 保留；新增 `grad_norm`（clip 前后）。
3. 阈值表 O1–O10 写成 run tags（如 `watch/gamma_sat=0.3`），evaluator 按 tag 复核，不依赖口头约定。
4. 不新增分支结构、不改模型前向签名；只加记录与诊断调用。
