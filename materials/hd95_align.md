# HD95 管线对齐验收（eval n-20261001-132706-9f0 ← impl n-20261001-111712-dbc, b6123d2）

口径依据：上游 `training/metrics.py:90 surface_metrics_single`（MONAI HD95 + 空回落 max_dim）与 `:40 postprocess_binary_mask`（最大连通域+填洞+去小目标+closing）。

## 1. 结论：通过（管线增益 ~2.3 HD95 确认；剩余 ~3.5 是模型差距）

三档本地独立重跑（dsfull ckpt，50 test 病人，n=500 帧帧帧非空）：

| 档 | cdist（自研） | mirror（MONAI） |
|---|---|---|
| A 裸（阈值 0.5） | 10.64 | 12.06 |
| C +后处理 | 8.29 | 9.01 |

- 后处理增益：−2.35（cdist）/ −3.05（mirror）✅；dice 0.9119→0.9146（+0.27pt，连通域去碎片）。
- mirror>cdist 恒成立（+1.42/+0.72）✅——anchor"以后报 mirror"正确（与上游对齐）；历史 10.64 口径声明更新为 mirror 12.06。
- 管线总增益 ~2.3（10.64→8.29 同口径）；剩余到论文 5.43 的 ~3.9（cdist）/ ~3.6（mirror）是模型差距 ✅（"it's the model"成立）。
- sweep：0.5→0.75 dice +0.21pt（0.9119→0.9140），10 点单调未饱和 ✅；阈值非瓶颈（0.2pt vs 后处理 2.3 HD95），与 anchor 一致。

## 2. 实现核验

- mirror：MONAI `HausdorffDistanceMetric(p95)` 直调 ✅；空约定对齐上游（双空 0/单空 max_dim；自研 inf/过滤）——单元验证一致；1px=1.0 双实现一致；随机 mask 对 1.0=1.0（非退化一致）。
- 后处理：照抄上游四件（scipy，默认关、eval 显式开）✅；噪声 mask 0.30→0.004（碎片清除有效）。
- OOM 广播 guard：dice 加 squeeze+assert ✅（61GB 先例）；sweep 侧 P/G 同形断言 ✅。
- 旗标：`--sweep/--postprocess/--hd95-mirror`（注意 mirror 旗是 `--hd95-mirror` 非 `--mirror`，已验证）。
- first/last-frame：后处理档 first 0.8786（裸 0.8605，+1.8pt）/last 持平 0.9076——边界增益集中在首帧（ED 侧形态差），记一笔。

## 3. 与论文对照（mirror 口径，CAMUS）

| | Dice | HD95 |
|---|---|---|
| 论文 DIAG | .9360 | 5.43 |
| 本次 C 档（后处理+0.5） | .9146 | 9.01 |
| 缺口 | −2.1pt | +3.6 |

- 剩余差距归属：未预训练基座 + λ 占位 + dts 恒 1 + 单次 2x 上采样（005 §1.4(b) 高分辨 decoder 未做）。下一步按 006 矩阵，不在此展开。
- sweep csv 落盘 `outputs/threshold_sweep.csv`（gitignored）✅；单调未饱和，0.75 以上未探（阈值非瓶颈，不追）。
