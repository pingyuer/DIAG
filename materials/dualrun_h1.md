# 双 run 验收：frame-only 基线 vs ds-full（eval n-20261001-064249-b14 ← impl n-20260930-185140-11d）

口径依据：`proposals/000_diag_paper_parse.md` §3–§4（表 1 CAMUS DIAG .9360/5.43；表 2 全配置 ED .9427/ES .9218/Mean .9317；frame-only 消融 −2.63pt）+ `proposals/003_ds_detach_svf.md` §4（加法顺序 ds→SVF→prototype，隔离增益，HD95 先崩即停）。

## 1. 结论：通过（带五条记录，不打回）

- 双 run 均 FINISHED、同 seed/split/budget（400/50/train_n，30ep×batch2，lr 3e-4，seed 0），曲线/ckpt/mlflow 逐项核对，父节点断言属实。
- 本地独立 spot-check（50 test 病人全量，正确 patient IDs）：ds-full test 0.9119/0.9119/10.64，frame-only test 0.8078/0.8078/40.30——与 val 方向一致。

## 2. H1 缺口（val 口径，主验收）

| | best val Dice | @ep | test frame-mean | test HD95 |
|---|---|---|---|---|
| frame-only（run 9ee3d1f8） | 0.8025 | 25 | 0.8078 | 40.30 |
| ds-full（run d5a3fd70） | 0.9168 | 29 | 0.9119 | 10.64 |
| 缺口 | **11.4pt**（val）/ 10.4pt（test） | — | — | — |

- 方向与论文一致（时序分支增益为正），量级更大（论文 Table 2 frame-only −2.63pt）。解释：随机初始化基座下内容锚弱，时序分支贡献被放大；frame-only 基线解码器直读 F_tf（P=F_tf、Q=0），容量亦小于全链。**禁止**将 11.4pt 直接引用为论文 −2.63pt 的复现。
- ds-full vs 首训（0.9086）：+0.8pt。**非纯隔离增益**：混杂 frame-only 分支代码变动（loss 调用、W 累加、ED/ES 块）与 ds_head/detach 引入，DETACH 与 ds 能量头贡献未分离。003 §4 要求的"隔离增益"仍缺（proposed `n-20260930-185336-dac` 属后继）。

## 3. HD95（Val 8 病例子集 / test 全量）

- ds-full：val 54.4→8.6；test 10.64（n=500）。frame-only：val 90.2→35.4；test 40.30。
- 时序分支边界优势明显（test 10.64 vs 40.30），但距论文 5.43 仍 +5.2（≈2×）。偏差延续：未预训练基座、λ 占位、dts 恒 1、自研 HD95 实现。

## 4. W-step 健康（240 点×双 run，独立拉取）

- ds 恒 1.0 全程（mean=min=1.0，240/240）：恒等延续成立、未塌零（`watch/ds_min=0.05` 无告警），但**什么也没学**——ds 头对训练零贡献、零风险。ds 终验（变间隔/缺失采样）待后继。
- K 0.515→0.886（ds-full）：校正门向观测侧漂移，O3 观察项（>0.95 阈值未触发，未达 200-step 持续判据的动作线，记观察）。
- gamma_sat 恒 0、gamma_mean 0.881→0.866：O1 安全，无饱和。
- best 直方图终态 j0/j1/j2=0.45/0.29/0.26：三候选分化，无 O4 垄断（阈值 0.95 远未触）。
- frame-only 的 k_mean=0/alpha=0 系分支占位（无 PCLF/HDC），非测量值；其 `gamma=0.88` 常数填充亦然——读窗口时须排除，不作健康判据。
- ds-full lr 触发 Plateau 两次降至 7.5e-05（ep18/28 后）；frame-only lr 恒定。调度器工作正常。

## 5. 产物与可追溯性

- ckpt md5 本地==远端（frameonly cd384a8f… / dsfull 7f735995…）；`frameonly_best.pt`/`dsfull_best.pt` 系重命名 copy（与 `best.pt` 同 md5），`last.pt` 同代。
- **ckpt 缺 `ds_head` 键**（双 ckpt keys 均无 ds_head/frame_only）：两 run 发起早于修复 commit 4cd55e2，落盘时代码无该键。ds_head 权重（1×96 Linear，仅 97 参数）未存档——可复算但不可复载，后继 run 已修。记为小债务，不打回（ds 恒 1.0，权重无信息量）。
- code_sha=bcd0fb59（双 run 同）==本地 `.code_sha`；tarball-sha 口径（见全链验收记录 C）。
- `eval_test.py` 无 `--frame-only` 旗标：本次 spot-check 用独立脚本（frame_only 分支内联）完成；后继 test 评估应给 eval 加旗标（与 anchor"test 评估待后继"一致）。

## 6. 记录修正：patient-avg 口径

- 全链验收记录 A 称 hash-ID 写法使 patient-avg 退化——本次用正确 `arange` IDs 重算：50 病人×10 帧等长下 patient-avg ≡ frame-mean（数学恒等，非 bug）。0.9119/0.8078 双口径一致，结论不受影响；原"无效"定级修正为"等长下恒等，数值有效"。
