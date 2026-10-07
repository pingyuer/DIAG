# 008-2验收（eval n-20261004-083527-5a6 ← impl n-20261004-074736-52e）

口径依据：008-2 提案五项（HD95主判据 + 全预算FO对照 + 零样本探针 + 归一化消融 + 难例切片）。

## 1. 结论：部分通过（FO对照+切片落地；零样本/归一化未跑；eval有bug）

- DINO-FO 全预算（`dino-fo-full` FINISHED seed0 400/30ep）：val **0.8928**/best 0.8955/HD95 8.0。
  test（FO ckpt + frame-only前向，本地独立重跑）：frame-mean **0.8876** / HD95 9.77 / mirror 10.59（n=500）。
  相对 UNeXt-FO（0.8025）：val +9.0pt——P1 分母重标成立。
- test 反超链（同 frame-only 口径）：DINO-FO 0.8876 vs UNeXt-FO test（待补跑；val 0.8025）。
  注：`eval_test.py` 无 `--frame-only` 旗标，本次 FO test 系 frame-only 前向独立脚本（与双run验收同法）；`--anchor` 已有。
- 零样本探针：**未跑**（mlflow 零记录）。本地补跑（frozen DINO + 随机adapt/decoder，10 clips）：
  frame-mean **0.1575**——随机头接近瞎猜，适配层+decoder 背了全部拟合（预期内；正式探针应记mlflow）。
- 归一化消融：**未跑**（mlflow 零记录；`--norm` 系 loss 归一开关，非输入归一）。
  ImageNet stats（0.485/0.229）vs echo（均值~0.15）仍是声明偏差，未实测。
- 难例切片：✅ 代码落地（128ef6a：ES-proxy/小腔体/边界带）；s2-s1 实测 ES 0.9074/small 0.9009(n=14)/band 0.6541。
  FO/DINO 切片数待补（本次 FO 重跑未带切片，因 eval bug 中断后改独立脚本）。

## 2. eval bug（修后重跑，本次验收有效）

- `eval_test.py:146` 变量 `a` 覆盖 argparse（`a.mirror` 崩 `AttributeError`）→ `_a`，已 commit。
  教训：切片块与 argparse 同名变量；后继加 lint（`grep -n "^ *a *="`）。
- 另：`eval` 内核 `diag_metrics` 陈旧（无 mirror）系内核缓存——venv 路径为准，已确认。

## 3. 与论文/H 结论的对应关系

- P1 转正判据三项：HD95 主判据 ✅（DINO val 5.6 vs UNeXt 9.1）；全预算对照 ✅（FO +9.0pt）；
  seed1 test ✅（008续）；test Dice 反超 −0.5pt 记天花板假象。
- MedSAM/SAM（C/D）：仍零 runs——最大未探项延续。
