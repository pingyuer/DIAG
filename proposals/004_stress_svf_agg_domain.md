# 004 提议：ds 压力 / SVF 转正 / patient 聚合 / 跨域试点 + 参数搜索纪律

依据：`proposals/001`（DPFR 差距）、`002`（W-step/W-epoch/W-event + O1–O10）、
`003`（ds/detach/SVF flag-off）+ 三份验收沉淀（`materials/fullchain_camus30.md`、
`materials/windows_002.md`、`materials/restructure_003.md`）+ 双 run 验收结论
（H1 缺口 11.4pt、ds-full +0.8pt、HD95 35 vs 8.6、ds 恒等 / K 漂移 / gamma 健康）。
本地实测补充：`src/diag/ds_head.py`（bias≈0.54/detach/clamp 1e-3/delta_scale=8）、
`src/diag/svf.py`（flag OFF、未接主链、S&S compose 数学存疑）、
`src/diag/data_camus.py`（dts=ones、ED/ES 首尾占位、四 DEVIATIONS）、
`DIAG-code/train_camus.py`（--frame-only 分支已落地、ds 接线 `ds_vec[0]` B=1、
W-step 含 ds_mean/ds_min）、`DIAG-code/eval_test.py:97`（patient 平均 hash bug）。
只整理、不实现。幂等核对：`proposals/` 下 000–003 存在，无 004，不重复。

## 1. ds 压力实验（Fig.5 式，ds 的终验）

目标：回答"ds 是否真学到时钟免疫，还是恒等 ones 的摆设"。
现状：`restructure_003` 记录 D 确认 `ds_mean=ds_min=1.0`（初值恒等延续，未塌零）；
`windows_002` 确认 K=0.50 安全区、gamma 健康。压力实验是 ds 第一次真考验。

设计（同一 ckpt，不重训，`eval_test.py` 式离线扰动）：
- P1 五帧：每 clip 均匀抽 5 帧，Dice 保留率（论文 ≥98.9%）。
- P2 三帧：均匀抽 3 帧，保留率（论文 95.9%）。
- P3 70% 随机缺失：每 clip 随机丢 7/10 帧后重组，保留率（论文 97.0%）。
- P4 40% 时间戳抖动：dts 乘 U(0.6,1.4) 噪声，保留率（论文 97.5%）。
- P5 ds 塌零检查：全程 log `ds_mean/ds_min` 分布；若某扰动下 `ds_min` 触 clamp
  下界 1e-3 占比 >10%，记 ds 饱和观察项（003 O-新增告警阈值 `watch/ds_min=0.05` 已写 tag）。
- P6 K 门漂移诊断：扰动前后 `K_mean` 对比；K→0（<0.05）=观测被弃，K→1（>0.95）=
  历史被弃，按 002 O2/O3 动作查 Δt 供给 / VectorField 范数 / smooth 权重。

注意：processed PNG 无真时间戳（`data_camus.py` DEVIATION），P4 的"抖动"是
人工乘子，不是真 wall-clock；结论写明此限，不冒充真时钟免疫。

## 2. SVF 转正（flag-off → on，条件转正）

前置：`restructure_003` §3 观察项——`exp(v)∘exp(-v)` 在 v~N(0,0.02) 下
maxdiff≈1.6（归一化坐标），疑似 compose 步位移加倍语义偏差。**转正前必须先修**，
否则免谈。

转正清单：
- T0 修 compose 数学：重验逆一致性，v=0.001 时 maxdiff 应 <0.05；修后恒等三件
  重过（OFF maxdiff=0.0 / `exp(0)` 恒等 / ON 初值位移 ≤0.05）。
- T1 转正标准（003 原样）：warp 分支 `flow_prompt_delta > 0` + 诊断三项
  （ds 不塌零 / 恒等 phi 下 Dice 不变 / `gamma_cv` 量级不恶化）。
- T2 超参：S&S 步数 {4,6,8} / `max_disp` {0.03,0.05,0.08} / SVF smooth 权重网格
  ——一律走 §6 搜索，不手调。
- T3 `align_corners=False` 全链统一（`svf.py` 已定，DPFR 用 True，禁止混用）；
  `identity_grid` 按 shape 缓存（已落地，保持）。
- 转正失败回退：任一标准不满足 → SVF 分支**删除**（按 003"否则删除，不堆 trick"），
  γ 通道门控 + H4 诊断保持不动。

## 3. patient 聚合修复（补 000 §5 缺失项）

现状三处占位（全部已在验收中标记）：
- R1 `eval_test.py:97`：`hash(p) % 10**6` 下 `patient_average` 退化为 frame-mean
  （两者均为 0.9132，全链验收记录 A）。修复：真 patient id
  `arange(50).repeat_interleave(10)` 重算 patient-avg Dice/HD95。
- R2 ED/ES：`val_ED=pv[[0]]`、`val_ES=pv[[-1]]` 首尾帧占位
  （`data_camus.py` ed_es DEVIATION + windows 记录 C）。修复二选一：
  (a) 沿用占位但改名 `first/last-frame`，禁称 ED/ES；
  (b) 按 CAMUS 官方 ED/ES 元数据对齐（需 human 给元数据路径）。
  在 (b) 到位前强制 (a)。
- R3 seed/split 锁死：官方 split json
  (`camus_public_datasplit_20250706.json`，400/50/50 无泄漏已验) + `torch.manual_seed(0)`；
  写死进 run tag（`seed/split_sha`），任何重跑先对 split md5。
- 口径对照表（imp 输出、evaluator 复核）：frame-mean vs patient-avg
  （Dice/HD95/时序三件套各一列），Table 2 式 CAMUS Mean 以 patient-avg 为准，
  在 R1 修复前禁止引用当前 0.9132 为论文对照。

## 4. 跨域试点（只出方案，不全量跑）

范围：Adult / Pediatric / CardiacUDA 的 processed 十帧 clip（`DATA` 同级目录，
`DIAG_CAMUS` 环境变量式覆盖已支持多 root）。
试点 gate（按序，任一不过即停）：
- G0 数据存在性：远端 processed 目录 + split json 齐 → 否则试点取消，不追数据。
- G1 同协议：prompt-free 推理、无 GT/box、`label_valid` 监督帧口径、
  foreground Dice 定义、logits 对齐到目标 mask 尺寸（沿用 upstream README
  §"intended fairness grain"，GDKVM-vs-DPFR 公平粒度）。
- G2 先试点 CardiacUDA（论文 DIAG 最优 .8092、GDKVM 掉到 .7279，区分度最大），
  再 Adult/Pediatric。只报试点 Dice/HD95 + 与 CAMUS 基线的 delta，不做 mDice
  （mDice 只报全覆盖方法，000 §3）。
- 全量四域对照是独立后继节点，本提议只到试点方案。

## 5. 架构选择（加法顺序，防 trick）

ds → SVF → prototype buffer，每个只看隔离增益 + 两两交互，无增益即删；
HD95 先崩即停（Dice 会骗人）；frozen 预训练 backbone + frame-only 基线对照
（003 §4，backbone 替换本身另起提议，不在本 004 内）。
当前已知锚点：H1 缺口 11.4pt（frame-only vs full，待 patient-avg 修复后重标）、
ds-full +0.8pt（ds 头隔离增益初值）、HD95 35 vs 8.6（边界路还远，O7 观察态）。

## 6. 参数搜索纪律（禁手调报喜）

凡多组实验（λ 六项权重 / lr / ds clamp 下界 / `delta_scale` / SVF smooth /
S&S 步数 / `max_disp`），一律走网格搜索：
- mlflow 记全网格（params 全量 + `code_sha` + split md5 + seed），单点手调值
  禁止作为结论引用。
- 先粗后细：λ 先按量级 {0.01,0.1,1.0} 扫（现状占位 ce=1/dice=1/rec=0.1/
  smooth=0.01/flow=0.1/iou=0.5），再细化有效维度。
- 搜索预算写死（max runs），超预算停；最佳点需独立 seed 复跑一次才转正。
- `loss/*_ep` 六项 epoch 均值（windows 记录 C 小缺口）随搜索一并补上，
  否则网格间不可比。

## 7. 给 implementation 的落地清单（按序）

1. R1+R2(a)+R3：patient-avg 真 id 重算 + 首尾帧改名 + split md5/seed tags
  （纯评估侧，不动训练）。
2. P1–P6 ds 压力：`eval_test.py` 式离线扰动脚本（新文件，不动训练链），
   同一 ckpt 出保留率表 + ds/K 诊断。
3. T0：SVF compose 修复 + 逆一致性重验（修不好就删分支，直接 T-回退）。
4. T1–T3 + §6 网格：SVF 转正搜索（远端跑，mlflow 全记）。
5. G0–G2：跨域试点 gate 检查 + CardiacUDA 先试点。
6. 不做：backbone 替换、独立原型池、associative scan（003 §5.5 原样）。
