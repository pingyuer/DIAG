# 单 clip 过拟合验收（eval n-20260930-151142-e95 ← impl n-20260930-145530-951, commit 394e030）

口径依据：父 spec（优化通路活：同一 clip loss 明显下降 + WTA Dice 上升；数值相对论文无意义）+ `proposals/000_diag_paper_parse.md` 口径（表 1–2 仍待训练后对照）。

## 1. 结论：打回（blocked）——缺 checkpoint，优化通路本身通过

- 优化通路 ✅：run `072c1454` FINISHED，60 步 loss 1.7678→0.4399、dice 0.0471→0.9527（与 anchor 声称 1.77→0.44 / 0.05→0.95 一致，逐点历史已核）。
- 打回项 ❌：spec 第 4 条"保存首个 checkpoint 到 outputs/（不进 git）"未落实——`train_overfit.py` 全文件无 `torch.save`，远端 `:31035 /root/DIAG/outputs/` 仅 `smoke_gpu.json`，本地 `outputs/` 无任何 `.pt`。"checkpoint 可追溯"的 Done 标准不满足。
- 另三 run（47ad27ec / e6d394c0 / cdea6f87）均为 FAILED、0 metric 点，系修 bug 前的调试迭代；有效 run 仅 072c1454。

## 2. 实测复核

| 断言 | 实测 |
|---|---|
| loss 下 + dice 上（同 clip） | ✅ 1.7678→0.4399 / 0.0471→0.9527，60 点连续（前 5 步含 0.0 抖动，方向成立） |
| asserts 通过（l1<l0, d1>d0） | ✅ run FINISHED 即断言通过 |
| 偏差 tags（dts/ed_es/lambda/backbone） | ✅ 四 tag 全在 run 上，与 `DEVIATIONS` 一致 |
| params（patient/steps/lr）+ code_sha 打标 | ✅ patient0001/60/3e-4，sha=727e8991（==本地 `.code_sha`==远端 `:31035`，同步） |
| losses 布尔索引修复（B=1/B=2 可反传） | ✅ 独立验证 B=1 partial-Omega 与 B=2 均可 backward；B=2 全 Omega 新旧路径数值恒等（旧代码"蒙混过关"成立，但仅全 Omega 下成立） |
| 主训练旧代码影响 | ⚠️ `train_camus.py` 逐病人循环 lab=(10,1) 全 True，旧索引恰好等价（已证），故 `:32237` 历史曲线数值不受影响；但 partial-Omega 语义已变，后继训练须用新 losses（与其 anchor 注记一致） |
| data 形状（(T,1,1,H,W)，逐帧即 BCHW） | ✅ 代码读验 + 与 anchor 修 bug 描述一致 |
| checkpoint 落盘 | ❌ 无 `torch.save`，远端/本地均无 `.pt` |

## 3. 打回要求（implementation 侧）

1. `train_overfit.py` 加 `torch.save`（模型 state_dict + `code_sha` + 步数/dice，落远端 `outputs/overfit_1clip.pt`）。
2. 远端重跑（~20s）或复用当次权重（已丢失则重跑），`sync.sh pull` 拉回本地 `outputs/`（gitignored，不进 git）。
3. mlflow run 保持 `node=train-overfit` + 同组 DEVIATION tags；新 run 替代 072c1454 为验收有效 run。

## 4. 与论文结论的对应关系

- 本节点只证"优化通路活"，不产生 Table 1–2 对照数值；dice 0.95 为单 clip 记忆，**禁止**引用。
- `:32237` 主训练 ep003 val_dice 0.83（anchor 提及）属另一 running 节点，其验收待该节点提交后另行 evaluation，本文件不覆盖。

## 5. 打回项关闭（重验 2026-09-30，重验 run overfit-1clip-ckpt 0b679c2a）

- `train_overfit.py` 已加 `torch.save`（commit 0a01047，diff 已读验：5 state_dict + loss/dice 起止 + code_sha，远端落 `/root/DIAG/outputs/overfit_1clip.pt`，本地 fallback 路径保留）。
- 新 run `overfit-1clip-ckpt` FINISHED：60 点 loss 1.7678→0.4454、dice 0.0471→0.9502（与旧 072c1454 同分布，方向一致）；四 DEVIATION tags + patient/steps/lr params 齐；code_sha=d6fcd138（当次 tarball）。
- ckpt `outputs/remote-31035/overfit_1clip.pt`（33MB，gitignored）：md5 本地==远端（43856bcd…），内容含 anchor/pclf/hdc/dec 四 state（参数量 7625314/406848/94274/128868 与各验收一致）+ loss/dice 起止 + sha。打回项关闭，本节点转 done。
