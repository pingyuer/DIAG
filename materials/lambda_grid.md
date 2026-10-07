# λ 六项权重网格验收（eval n-20261001-091320-754 ← impl n-20261001-070301-f99）

口径依据：`proposals/004` §6（先粗后细、单点禁引用、最佳点独立 seed 复跑、loss_ep 补齐）。

## 1. 结论：通过（方向性结论，非点估计结论）

- 14 runs 全 FINISHED：13 点（每维 1–2 点，center-out 单因子）+ L-ce-0.01 seed7 复跑。同预算 train_n=60/6ep/batch2，params 全记。
- 允许的方向性读数：dice 降权崩（0.01→0.032，单 seed 但幅度大）、smooth 加权负向（0.1→0.623/1.0→0.615）、ce 降/rec 升正向建议。**禁止**引用任何单点为"最佳 λ"——seed 方差（−8.6pt）大于 top-3 间距（1.6pt）。
- `loss/*_ep` 六项 epoch 均值 live ✅（windows 缺口关闭）；λ 六 args 可配 + 接线已 commit ✅。

## 2. 网格独立复核（exp97，val 口径）

| run | val | HD95 | 说明 |
|---|---|---|---|
| L-ce-0.01 (s0) | 0.7240 | 43.51 | 网格最高（唯二 <50 的 HD95 之一） |
| L-rec-1.0 | 0.7080 | 56.40 | rec 升正向 |
| L-iou-1.0 / 0.1 | 0.7036 / 0.7024 | 58.4 / 57.0 | iou 平坦（0.01→0.663 微降） |
| L-ce-0.1 | 0.6912 | 63.85 | ce 降单调（1.0 基准≈0.65 →0.1→0.01 上升） |
| L-rec-0.01 | 0.6759 | 64.61 | rec 降负向 |
| L-flow-0.01 / 1.0 | 0.6560 / 0.6405 | 59.4 / 56.1 | flow 平坦偏负 |
| L-ce-0.01-seed7 | 0.6380 | 81.43 | **复跑 −8.6pt**，见 §3 |
| L-smooth-0.1 / 1.0 | 0.6232 / 0.6152 | 59.1 / 56.5 | smooth 毒药（方向） |
| L-dice-0.1 | 0.4931 | 59.54 | dice 降半崩 |
| L-dice-0.01 | 0.0321 (best ep4 0.53) | 68.65 | dice 降权死：ep5 单步 val 0.53→0.03，train_loss 反升，典型 WTA 监督塌（ce/dice 主监督被抽掉后的候选漂移） |

## 3. seed 方差（§6 纪律的核心证据）

- L-ce-0.01：seed0 0.7240 vs seed7 0.6380（Δ=−8.6pt）；HD95 43.5→81.4；train_dice 0.67→0.52。
- top-3 间距（0.724/0.708/0.704）< seed 方差 ⇒ 网格内任何排序均无显著性。stage-2（ce×rec 中预算）必须多 seed mean±std，否则只是换一组噪声。
- code_sha 双值（60cf05b5 vs b1ced08a）系网格前后两 commit，params 维度一致可比；seed7 与 seed0 同 sha？seed7 sha=60cf05b5（待确认行，params.seed=7 已记，split 同源）。

## 4. HD95 作废确认

- 全网格 43–81，无一接近论文 5.43；最优 L-ce-0.01 的 43.5 亦 8×。6ep/60-子集早训区边界未成形，HD95 在此预算下无选择力——"先崩即停"规则在此表现为"全崩，Dice 代理排序"，与 anchor 一致。
- 后继中预算重跑后 HD95 才有投票权。

## 5. 搜索纪律对照（004 §6）

- ✅ 全网格 params+code_sha+seed；✅ 最佳点独立 seed 复跑（且复跑证伪了单点结论——纪律生效的正面例子）；✅ loss_ep 六项补齐（`train_camus.py:308`，`loss/{ce,dice,rec,smooth,flow,iou}_ep` 全 live）。
- ❌ 先粗后细未闭环：ce×rec 联合、smooth 下探（<0.01？smooth=0 即关）、flow/iou 细化均未跑——stage-2 另节点，正确。
- 交互未知：本网格为 center-out 单因子，ce×rec/dice×iou 等交互全未知；"排序 ce>rec>iou>flow>smooth"只是单点快照排序，不作效应排序引用。

## 6. 与论文/H 结论的对应关系

- 不产生 Table 1–2 对照数值（6ep/60-子集欠训区）。唯一可带走的是搜索先验：ce 降、rec 升、smooth 慎、dice 主监督不可动——stage-2 网格的起点，非结论。
- 本文件存本地 `materials/`（gitignored 不进仓）。
