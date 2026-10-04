# 006 提议：后处理 / TTA / 阈值 / 插值 / 时序平滑 / 扰动探针离线评估

依据：`materials/hd95_align.md`（三档先例：A 裸 10.64 / C 后处理 8.29 cdist，
mirror 12.06/9.01；后处理增益 −2.35 cdist / dice +0.27pt；阈值 0.5→0.75 +0.21pt
未饱和；剩余到论文 5.43 的 ~3.6 是模型差距）+ `DIAG-code/eval_test.py` 现状
（`--postprocess` 全开/全关 only、`--sweep` 在 test 侧 0.30–0.75 step 0.05、
无 flip-TTA/时序平滑/噪声探针代码）+ `DIAG-code/diag_metrics.py:192`
（`postprocess_binary_mask` 四开关已落地：largest/fill-hole/remove-small/closing
+ min_size=16）+ `materials/eval_008b2.md`（DINO-FO test 0.8876、切片已落地
ES-proxy/小腔体/边界带）。
只整理、不实现。幂等核对：`proposals/` 下 000–005、006_decoupling_matrix、
007_pipeline_cleanup 存在，无 006_post_eval，不重复（与 006_decoupling_matrix
正交：本提议纯离线 eval，零训练）。

判据统一：同 split（md5 `682f3d89`，eval 打印 `split_md5`）、裸口径看模型、
应用口径看表、单次只记读数。全离线 eval（CPU 可跑、零训练），固定 ckpt：
`outputs/grid1_s0_best.pt`、`outputs/grid1_s1_best.pt`、
`outputs/s2full_s1_best.pt`、`outputs/dino_s_full_best.pt`
（grid1/s2 为 UNeXt 系，dino 为 DINO 系；s0/s1 双 seed）。

## 1. 后处理逐项开关（新活：只做过裸 vs 全开）

- 现状缺口：`hd95_align` 只有 A 裸 / C 全开两档；`postprocess_binary_mask`
  四开关 + min_size 从未单项拆过。上游默认参数（min_size=16、全开）不预设为
  对齐目标——逐项实测后再定应用口径。
- 设计：固定 ckpt × {裸} ∪ {largest, fill-hole, remove-small, closing} 逐项
  on/off（2⁴=16 格，可先跑 4 项单开 + 全开共 5 格，再按需补交互格）+
  min_size {8, 16, 32, 64} 单维扫（其余三项全开）。
- 记数：每格 Dice（frame-mean + patient-avg）/ HD95 双口径（cdist + mirror，
  `--hd95-mirror` 已有）+ ES-proxy / 小腔体 / 边界带切片（`eval_test.py:132`
  已落地）。
- 落地：`eval_test.py` 加 `--pp-item largest,fill_holes,... --pp-min-size N`
 （透传四 bool + min_size），不动训练链。

## 2. hflip-TTA（新活，零代码）

- 设计：同 ckpt，identity vs flip-average（单帧内水平翻转→前向→翻回→与原
  logits 平均），无未来泄漏，因果安全。A4C 翻转仍是合法 LV 形态。
- 分基座记增益：UNeXt 系（grid1-s0/s1、s2-s1）与 DINO 系（dino-s0/s1）各记
  ΔDice / ΔHD95；验 DINO 零增益假设（frozen 特征翻转等变强，预期增益 ~0）。
- 落地：eval 加 `--tta-flip`（只改 eval 前向，不动训练）；TTA 只用于应用口径，
  裸口径永不掺 TTA。

## 3. 阈值细扫（方法修正：现有 sweep 在 test 侧）

- 现状缺口：`outputs/threshold_sweep.csv` 是 test 侧 0.30–0.75 step 0.05
  （`eval_test.py:181`）= 乐观偏差；val 择阈从未做过。
- 设计：val（50 病人）上 0.30–0.75 step 0.01 搜最优阈（Dice 主判据，HD95
  次判据），锁阈上 test 报数；分基座记（UNeXt vs DINO 最优阈可能不同）。
- 前置检查：val 数据本地是否齐（train 只用 val_ids[:8] 子集做 W-epoch）。
  不齐则此项记阻塞声明，不拿 test 当 val 使。
- 落地：`eval_test.py --sweep` 加 `--sweep-split val/test --sweep-step 0.01`，
  csv 落盘 gitignored。

## 4. mask 插值口径（验证 + 文档项为主）

- 现状：mask 侧 GT resize 全 nearest（train/eval 一致）；bilinear 只碰特征
  （decoder/hdc 上采样），未碰 mask。decoder 迟上采样位置不动。
- 设计：nearest vs bilinear 在 mask resize（pred→GT 对齐、GT→loss 对齐两处）
  各跑一遍对照，记 ΔDice / ΔHD95；预期差值 ~0（口径留痕，非增益项）。
- 高分辨 decoder（005 §1.4(b)）另立建模项，不混进后处理——本项只做口径对照。

## 5. 因果时序平滑（新活；与高斯场探针划界）

- 设计：过去窗 logits 平均（k=2,3）vs 单帧，严格不向未来看（t 只平均
  t−k+1..t）。只做 eval 侧平滑，不动训练、不进模型。
- 风险：ED/ES 跳变处滞后——必须单列 ES-proxy / 小腔体 / 边界带切片看滞后惩罚；
  full-mean 0.91 天花板下 headline 差值小，切片才是判据。
- 划界：高斯场是格 4 式因果探针（建模项），不是后处理；本项是纯 eval 平滑，
  两者不互认增益。
- 落地：eval 加 `--temporal-average k`（k=1 为恒等对照）。

## 6. 高斯扰动鲁棒探针（可选；只探鲁棒，不进表）

- 设计：输入级加噪（σ 扫 0.01–0.05）敏感度曲线；潜态加噪需 hook，先做输入级。
- 只记 mlflow / csv，不进主表；掉分斜率用于鲁棒声明，不做任何"改进"引用。

## 7. 给 implementation 的落地清单（按序，全离线）

1. `eval_test.py`：`--pp-item/--pp-min-size`（§1）→ 先跑 5 格（裸 + 4 单开）。
2. `--tta-flip`（§2）→ 分基座记增益。
3. val 数据齐性检查 → 通过才做 §3 val 择阈（0.30–0.75/0.01），锁阈上 test。
4. §4 插值对照（nearest vs bilinear，两处各一遍，留痕）。
5. `--temporal-average 2,3`（§5）+ 切片滞后表。
6. §6 输入级噪声曲线（可选，mlflow only）。
7. 不做：高分辨 decoder（另立项）、高斯场探针（建模项）、任何重训。
