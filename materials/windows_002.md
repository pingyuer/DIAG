# mlflow 三层记录窗口验收（eval n-20260930-183357-9c8 ← impl n-20260930-145928-48e, commit 7d62564）

口径依据：`proposals/002_mlflow_windows.md` §1（三层窗口定义）§2（O1–O10 观察–调整表）§4（落地清单：最小改动面、不改前向签名）。

## 1. 结论：通过（带三条修正/约束记录，不打回）

- W-step / W-epoch / W-event 三层、O1–O10 watch tags、win-smoke2 run——父节点断言基本属实；逐项独立核对见 §2。
- 记录 A（计数口径修正）：anchor 称"24 窗口指标全 live（wstep 16 + wepoch 7 + diag 1）"，独立核对为 **30 live 指标键**（wstep 16 + epoch 旧 5 + wepoch 新 7 + diag 1 + best_val_dice 1；旧 `train_loss/train_dice/val_loss/val_dice/lr` 5 项仍在记）。16/7/1 三组数本身正确，总数应为 30 而非 24。
- 记录 B（wstep/loss_total 定义）：`w_log` 的 `wstep/loss_total = ce均值 + dice均值`（仅两项之和），与 `DiagLoss.total`（六项加权）**不是同一量**。作"主损失趋势代理"可用，但禁止与 total 混比；后继若需真 total 的 step 曲线，应直接记 `out["total"]`。
- 记录 C（ED/ES 占位延续）：`val_ED=pv[[0]]`、`val_ES=pv[[-1]]` 为首尾帧占位（与 data_camus 的 ed_es DEVIATION 一致），非论文 ED/ES 定义；Table 2 式对照仍不可用。`val_hd95=89.6` 为 2ep 早期随机网数值（合理）；`diag/gamma_cv_val=3.7e-4` 为未训练值（距 1e-6 目标 ~370×，O8 观察项，合理）。

## 2. 实测复核（run win-smoke2 f1377282，subset=4/epochs=2/w_step=2/diag_every=1，FINISHED）

| 断言 | 实测 |
|---|---|
| W-step 16 键（loss×6+k/gamma/sat/alpha×2/grad×2/best×3） | ✅ 全 live，gstep 轴 4 点（2ep×2 clips/w_step=2） |
| 六项拆分 `loss_{ce,dice,rec,smooth,flow,iou}` | ✅ live；末值 ce=.182/dice=.597/rec=.094/smooth=.006/flow=.090/iou=.691 |
| `grad_post` 恒 1.0（clip 生效） | ✅ 4 点全 1.0；`grad_pre`=2.57（确被 clip 压住） |
| `best_j0=1.0` 垄断 | ✅ O4 观察项（2ep 早期未分化，与 anchor 注记一致）；`best` 直方图累积逻辑读验正确 |
| `k_mean`=0.50（O2/O3 安全区内） | ✅ live |
| `gamma_sat`=0.0、`gamma_mean`=0.881（初值，O1 安全） | ✅ live |
| `alpha_c/f`≈0.119（初值） | ✅ live |
| W-epoch 7 键（hd95/drift/rough/ifvar/ED/ES + loss_ep 缺） | ⚠️ `loss/*_ep` 六项 epoch 均值（002 §1 要求）未落地——`tl/td` 只进了旧 `train_loss`，六项拆分的 epoch 均值缺。清单 §4 第 1 条部分未完成，记为小缺口（W-step 拆分已覆盖同类信息，不打回） |
| W-event 诊断 `diag/gamma_cv_val` 每 K epoch | ✅ 2 点（ep0/1，diag_every=1） |
| watch/×8 tags + args 全量 params | ✅ 8/8 无缺失；params 含 w_step/diag_every/subset/train_n/val_n |
| 前向签名未改 | ✅ `git diff f9398c5 7d62564 -- src/` 空；三处 rank bug 修只在 `train_camus.py` 内（loader 加 channel 维、gt_interp 去 unsqueeze、HD95 去 batch 维） |
| 旧 30ep run 无窗口 | ✅ 确认（历史局限，新 run 才有；全链验收结论不受影响） |

## 3. 与验收挂钩（002 §3，H1–H4）

- 窗口链路（W-step 前兆 + W-epoch 双轨 + W-event 诊断快照）已具备承载 H1–H4 实证的条件；数值本身（val_dice=0.35、HD95=89.6、CV=3.7e-4）为 2ep smoke 快照，**禁止**引用为论文对照。
- O1–O10 阈值已写死为 run tags（evaluator 按 tag 复核成立）；O4（best 垄断）当前触发观察态，O8（CV 高 370×）为未训练预期态——两者均为"观察项"而非"故障项"。
- 缺口汇总：`loss/*_ep` 未落地（小）；`state_intervention_drop` / 复合误差 / 采样压力快照未进 W-event（002 §1 要求，当前仅 gamma_cv；中缺口，待后继）；ED/ES 占位（已知偏差）。
