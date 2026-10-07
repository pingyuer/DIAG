# 004 验收（eval n-20261001-070109-f87 ← impl n-20261001-064846-e90, commit ae13427）

口径依据：`proposals/004_stress_svf_agg_domain.md` §7 落地清单（R1/R2a/R3、P1–P6、T0、G0–G2）。

## 1. 结论：通过（带六条记录，不打回）

- R1/R2a/R3、P1–P6、T0、G2 四组全部独立复核；anchor 断言基本属实，数值偏差见记录。

## 2. R1–R3（评估侧，本地重跑 `eval_test.py` 全 50 test 病人）

| 输出 | 重跑值 | anchor | 判定 |
|---|---|---|---|
| frame-mean | 0.9119 | — | ✅ 一致（双 run 验收 0.9119） |
| patient-avg（真 id） | 0.9119 | 0.9119 | ✅ |
| first-frame | 0.8605 | 0.86 | ✅ |
| last-frame | 0.9076 | 0.91 | ✅ |
| split_md5 / seed | 682f3d89 / 0 | — | ✅（R3 落地） |
| HD95 / drift / rough / var | 10.64 / 57.74 / 138.0 / 0.0431 | — | ✅（与双 run spot-check 同分布；drift/rough 与首训 51/119 量级一致） |

- 记录 R1：patient-avg==frame-mean 系等长 10 帧/病人的数学恒等（双 run 验收 §6 已修正），非口径 bug；真 id 落地后该行输出有效。
- 记录 R2a：ED/ES 已改名 first/last-frame ✅；官方 ED/ES 元数据仍缺（human 待给）。
- 小瑕：`per_pid` 变量保留未用、`patient_average` import 残留——死代码，未用即无错，不打回。

## 3. P1–P6 ds 压力（run `ds-stress-p1p6`，同一 dsfull ckpt，15 clips）

| | run 值 | 论文 | 判定 |
|---|---|---|---|
| retain P1（5 帧） | 0.9755 | 0.989 | ✅ 方向一致，−1.3pt |
| retain P2（3 帧） | 0.9421 | 0.959 | ✅ −1.7pt |
| retain P3（70% 缺失） | 0.9424 | 0.970 | ✅ −2.8pt |
| retain P4（40% 抖动） | 1.0000 | 0.975 | ✅ 超论文（见记录 P4） |
| P5 ds_floor_ratio | 0.0 | 告警 >0.10 | ✅ 过（ds 恒 1.0 未塌零） |
| P6 k_drift | −0.0052 | 漂移判据 | ✅ 过（jitter 不弃历史） |

- 独立 3-clip spot-check：base≈0.936，retain_p1 0.98–1.00，retain_p4 1.001–1.003（含>1 噪声上浮），ds_min=1.0，K≈0.886→jitter 0.876–0.891——与 run 一致。
- 记录 P4：P4 实现绕过 ds 头（`pclf.forward(..., jit)` 直接喂 ones×抖动，stress_ds.py:110），测的是 PCLF dt 鲁棒性而非 ds 时钟；且 ds 恒 1.0 下"时钟免疫"无从谈起。P4≈1.0 只证 Euler 残差更新对 dt 乘子不敏感，不证 ds 学到 anything。
- 记录 P5：ds 恒 1.0 通过 P5 是"未塌零"的弱通过；`watch/ds_min=0.05` 全程静默。ds 终验（变间隔真值）仍缺——PNG 无 wall-clock（提议 §1 已自限）。
- 记录 run 重复：exp97 内 `ds-stress-p1p6` 有两行同值 run（c82e7406 FINISHED + 7314b6b8 曾 RUNNING 同值），系脚本重跑/僵尸行；以后者 FINISHED 为准。同 code_sha=bcd0fb59（双 run 世代）。
- ckpt 缺 `ds_head` 键（双 run 验收 §5 债务延续）：stress 脚本 `WARN` 分支覆盖，ds 取初值 1.0——与训练时一致（训练落盘前 ds 即初值态），结论不受影响。

## 4. T0 SVF compose 修复（`src/diag/svf.py`）

- 新实现（位移自复合 `u <- u + u(id+u)`）fresh 解释器验证：`exp(0)` 恒等 maxdiff=0.0 ✅；逆一致 scale=0.02/0.005/0.001 下为 0.28/0.075/0.015（旧实现 1.61/0.94/0.20）——约 6–13× 改善 ✅。
- anchor 称"0.19→0.0002"：0.19 疑为旧代码在某小尺度下的值（旧 0.001 尺度我测 0.196，吻合），0.0002 量级在更小位移/初值 regime（ON 初值位移 ~1e-4）下合理。方向与量级确认，确切两数字的测量条件未留痕——记为"改善确认，端点数字条件不明"，不打回。
- 恒等三件（OFF maxdiff=0.0 / exp(0) / ON 初值 ≤0.05）✅ 全过（003 验收既有 + 本次重验）。
- SVF 转正搜索（T1–T3 网格）未跑 ✅（与 anchor 一致，留后继；分支仍未接主链）。

## 5. G2 CardiacUDA 试点（run `cardiacuda-g2-pilot`，dice=0.0394/HD95=75.05/n_lab=157）

- G0 ✅：远端 `/input3/processed/cardiacuda_a4c_lv_png128_10f/test/{img,label,metadata}` 齐；29/29 metadata 的 `label_indices` 均为 list（稀疏标注，每 clip 5 帧）。
- G1 ✅：forward 无 GT（仅 loss target 处用 label）；`label_indices` 过滤计分；128px 下 decoder 输出 128 与 label 对齐（含 shape 兜底插值）。
- G2 负结果确认：本地单 clip 探针——输入均值 0.079（CAMUS 0.153，约一半亮度）、128 vs 256 分辨率、pred fg 2.1% vs gt fg 2.9%（欠激活非全空）、logits 均值 −14.3（系统性负偏）。零样本崩归因（暗输入+分辨率 gap+无归一化适配）成立，按 G-gate 停止正确，不追全量 ✅。
- 数据 pull 本地 `outputs/cardiacuda_pilot/{img,label,metadata}` 29 clips（gitignored ✅）。

## 6. 与论文结论的对应关系

- P1–P4 保留率只报"同一 ckpt 扰动保留率"，相对论文 Fig.5（训练态 DIAG）−1.3~−2.8pt；未预训练基座+占位 λ 下的方向一致性证据，**禁止**引用为 Fig.5 复现。
- G2 的 0.039 系零样本跨域崩溃值，与论文 CardiacUDA .8092（域内训练）不可比；仅作 gate-stop 依据。
- 缺口延续（均未在本节点解决）：λ 真值、预训练基座、官方 ED/ES、SVF 转正搜索、ds 真间隔终验。
