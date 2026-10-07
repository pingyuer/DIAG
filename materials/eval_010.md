# 010验收（eval n-20261005-122826-d51 ← impl n-20261005-111048-7d6 ← proposal 010）

口径依据：`proposals/010_paper_compare.md` §5 + 009 §7（舱外数标引用、禁并表；并表只许C1–C4+R0-6同协议行）。

基线=a4c11e7零漂移（`git diff --stat`空）。upstream_BanditPM是独立git仓（`c507adc`，DIAG侧untracked只读对照，零修改）。

## 1. C2 GDKVM fair双seed：真跑FINISHED，双seed差0.19pt ✅

exp98 `gdkvm-compare`（mlflow直读全run_id，非前缀8位）：

| seed | run_id | test Dice/IoU | val Dice | 协议 |
|---|---|---|---|---|
| s0 | `3049b2121fe54c1592de2c894a8ba89d` | 0.9337/0.8771 | 0.9331 | camus_short_dense，400/50/50，10帧，256px |
| s1 | `96f4aecb865341239f37671996c5deff` | 0.9318/0.8746 | 0.9330 | 同上 |

- 双seed内差0.19pt（test）/0.01pt（val）——远小于R0阈值1pt ✅，无FLAKY。
- provenance：tag `git_commit=c507adc` + dirty=False + repoURL=BanditPM + `protocol=camus_short_dense` ✅；`use_first_frame_gt_init=False` + `init_mode_train=pred_or_zero` + `backbone_pretrained=False` ✅（与DIAG prompt-free对齐，010最大协议差已关）。
- split：`split_count/train400/val50/test50` + `path=/input0/processed/camus_png256_10f` + `seq_length=10` ✅（与R0同数据）。
- C2 B表回填：test Dice 0.9337/0.9318 ✅；HD95维持"待"（upstream只报Dice/IoU，无HD95——B表"待"字保留，非缺失）。

## 2. C3 DPFR-fair：未跑，维持引用

- 006既有test 0.9348维持"论文引用数（协议未对齐）"，不进B表 ✅（009 §7.2规则）。
- 队列pending残留2 task（c3-gdkvm-s1 + c2-gdkvm-s0）是**重排占位task**（与已FINISHED的exp98双run同名配置），非待跑新活——C2已出数，无需重跑（重跑即浪费卡）。

## 3. compare_table核验：A表引用 + B表回填一致

- node out与materials版diff空 ✅（同文件）。
- A表7行：3D U-Net .9162/5.46、ConvLSTM .9314/3.00、GDKVM .9368/6.13、MemSAM .9284/–、OSA .9284/5.83、SAM2-memory .9212/6.79、DIAG论文行 .9360/5.43——全标引用，无并表 ✅。
- B表：C1-FO双行 + DIAG s2双seed + R0-6 s1（s0 FLAKY不进表）+ C2双seed回填（本节点）+ C3待——与009/R0口径一致 ✅。
- wiki：`docs/compare_table.md`已同步（rebuild产物），nav有"舱内外对照" ✅。

## 4. 与论文/H结论的对应关系

- 不产生Table对照数值（对照表是定位尺，不是 verdict）。
- 定位：GDKVM fair 0.9337（Eq.1共享状态实例）vs DIAG R0-6 s1 0.9185——差1.5pt，方向是"共享状态+prompted初始化"仍强于我方prompt-free因果链；缺口归因不变（基座/λ/上采样/MedSAM阻塞）。
- C3后继：DPFR-fair同口径复核仍是open项（六提案草案/009 §6队尾）。
