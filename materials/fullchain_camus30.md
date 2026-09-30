# CAMUS 全链训练验收（eval n-20260930-182638-4a6 ← impl n-20260930-145330-a85）

口径依据：`proposals/000_diag_paper_parse.md` §3（表 1：CAMUS DIAG .9360/5.43、自动块、prompt-free；mDice 只报全覆盖方法）+ impl anchor 断言（30ep / best 0.9086@ep28 / test 0.9132/9.79）。

## 1. 结论：通过（带三条修正/约束记录，不打回）

- 30ep 收敛曲线、best.pt 内容、mlflow run、test frame-mean/HD95——父节点断言全部属实，本地独立重跑复现（frame-mean dice=0.9132、HD95=9.79、n=500、var=0.0443逐值一致）。
- 记录 A（patient-avg 无效）：`eval_test.py:97` 的 `hash(p) % 10**6` 写法下，`patient_average` 退化为 frame-mean（实测两者均为 0.9132）。`patient_average` 本体经单元验证正确（metrics 验收），错在调用方 ID 构造。真正的 patient-avg 须用 `arange(50).repeat_interleave(10)` 重算——本节点不代算，只标记该行输出无效。表 2 式的 CAMUS Mean 对照以此为前提，当前仍缺有效值。
- 记录 B（ckpt 世代错位）：本地 `outputs/remote-32237/camus_train/best.pt`（md5 dc64…）≠ 远端现存 `best.pt`（md5 807d…，epoch=1/val=0.3468/sha=7076d57a）。远端训练目录在 harvest 后被新一轮 2ep smoke（run win-smoke2 f1377282，sha=当前 HEAD）覆盖；本地 pull 物是 30ep 世代（epoch=28/val=0.9086/sha=4cc93ede），与 train.log（LOG_IDENTICAL）及 mlflow run 自洽。验收对象是本地 pull 世代 + run f4c358c7，远端现存文件不作为证据。
- 记录 C（code_sha 性质）：`4cc93ede…` 是 tarball sha256（非 git object，`git cat-file` 无此对象符合预期），ckpt 内 `code_sha` 与 mlflow tag 一致；`weights_only=False` load（ckpt 含 optimizer state，必须）。

## 2. 实测复核

| 断言 | 实测 |
|---|---|
| 30ep 曲线（train_loss 1.02→0.45 / val 0.80→0.47 / train_dice 0.49→0.92 / val 0.70→0.91） | ✅ train.log 30 行，首尾与 mlflow 4 曲线一致；mlflow 30 点×4 + lr 恒 3e-4 + best 16 点 |
| best_val_dice=0.9086@ep28 | ✅ ckpt epoch=28/val=0.9086；ep22 后 plateau 肉眼确认（0.906x 徘徊，ep26 回落 0.8991 后回升） |
| 收敛后 3ep（ep26–28） | ⚠️ ep26 train/val 双升（loss+0.014/+0.019，dice−0.01/−0.008）后回落，属小 batch 噪声，无发散；lr 恒定符合"Plateau 未触发"注记 |
| test frame-mean 0.9132 / HD95 9.79 / drift 51 / rough 119 / var 0.044 | ✅ 本地重跑逐值复现（仅 rough 119.1 vs 119 取整差） |
| split 无泄漏（400/50/50 不交） | ✅ tr-va/tr-te/va-te 交集全 0 |
| test 数据 50 病人×10 帧本地齐 | ✅ `outputs/camus_test_pull` 50 目录（gitignored，不进 git） |
| 产物 gitignored | ✅ `camus_test_pull` + `best.pt` 均 IGNORED |
| B=2 全 Omega 旧 losses 等价 | ✅ 沿用过拟合验收结论（数值恒等已证），曲线有效 |

## 3. mlflow run（exp97 `diag-camus-train`，run f4c358c7 FINISHED）

- tags：`node=camus-fullchain-train` + backbone/lambda/dts 三 DEVIATION；params train_n=400/epochs=30/batch=2/lr=3e-4。
- 无 artifact：ckpt（39MB×2）未 `log_artifact`，仅靠 pull 目录追溯。当前可接受（pull 物 md5 已记），后继训练建议加 `log_artifact` 或至少记 md5 tag。
- 同 exp 另有 smoke-train-4patients ×2、overfit ×4（均见过拟合验收）、win-smoke ×3 + win-smoke2 ×1（新一轮，属 002 落地节点范围，非本节点证据）。

## 4. 与论文结论的对应关系（CAMUS 列，Dice↑/HD95↓）

| | Dice | HD95 |
|---|---|---|
| 论文 DIAG | .9360 | 5.43 |
| 本次 test（best.pt，frame-mean，未训练基座+占位 λ） | .9132 | 9.79 |
| 缺口 | −2.28pt | +4.36（≈1.8×） |

- 口径对齐：prompt-free 推理（无 GT/box）✅；patient-wise 划分 ✅（官方 split）；十帧/视频 ✅。
- 口径偏差（解释缺口用，不作失败判据）：① 未预训练基座（UNeXt 随机初始化，记偏差）；② λ 占位；③ dts 全 1；④ frame-mean 代 patient-avg（记录 A，HD95/Dice 均受影响，patient-avg 待重算）；⑤ HD95 为自研边界点集实现（统一 surface 口径待补充材料）；⑥ val 选 checkpoint（0.9086）与 test（0.9132）差 +0.26pt，泛化方向正常。
- mDice / H1–H4：单数据集训练，无跨域结果，mDice 不适用；Table 2 消融、Fig.3–5 诊断均未跑——待后继节点，**禁止**将本表引用为论文对照结论。
