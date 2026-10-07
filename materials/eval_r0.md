# R0验收（eval n-20261004-153945-af6 ← impl n-20261004-140705-efc ← proposal 009）

口径依据：`proposals/009_dino_baseline_lock.md` §5（R0复现批 + FLAKY判定：同ckpt重跑差>0.5pt或同配置双seed内差>1pt→FLAKY不进锁定表）。

基线=ec71ccf零漂移（`git diff --stat`空）。R0-6 s1 ckpt sha `569e0424`是tarball-sha（非git object，`git cat-file`无此对象符合预期，与既有code_sha口径一致）。

## 1. R0-1~R0-5：同ckpt幂等 + 双seed内差，全过

r0_offline.log本地重跑复核（6 ckpt，test50/n=500，split全682f3d89）：

| 格 | s0 test裸 | s1 test裸 | 双seed内差 | 判定 |
|---|---|---|---|---|
| R0-2 grid1 | 0.9110/11.00 | 0.9116/10.54 | 0.06pt | ✅锁定 |
| R0-3 s2 | 0.9167/9.39 | 0.9140/11.14 | 0.27pt | ✅锁定（注：s1 HD95 11.14反跳，Dice仍+0.3pt，HD95方差记一笔） |
| R0-5 DINO-full | 0.9082/7.32 | 0.9085/7.65 | 0.03pt | ✅幂等锁定 |
| R0-4 DINO-FO | 0.8880/9.74（s0重跑，原0.8876差0.0004幂等内） | s1未跑 | — | ✅分母落定（s1待补，同R0-1 s1） |
| R0-1 UNeXt-FO | 沿用0.8078/40.30（旧世代ckpt不在outputs/） | s1未跑 | — | 沿用，s1待补 |

- R0-1/R0-4的s1补跑未做（FO-ckpt旧世代/单seed）——记未跑，不拦v1.0（分母方向已定+9.0pt，双seed落定待后继）。
- baseline_v10.md草稿§21"R0-1~R0-5最大0.27pt"与实测一致 ✅。

## 2. R0-6（a1，唯一新训练）：s1新最优，s0标FLAKY候选

mlflow双FINISHED（同budget 400/30ep，anchor=dinov2-small，ce=0.05/rec=1.0，vf_deep=True）：

| seed | run | best_val | test裸Dice/HD95 | ckpt md5前8 |
|---|---|---|---|---|
| s0 | a80cc896 | 0.9175 | 0.9066/7.66（本地重跑复现一致） | bcfca864 |
| s1 | 8a1e95d1 | 0.9224 | 0.9185/6.97（本地重跑复现一致） | daeea123 |

- 双seed内差：val 0.49pt / test **1.19pt**——超009阈值1pt → **s0标FLAKY候选，不进锁定表**；v1.0以s1为a1代表值（val 0.9224/test 0.9185/HD95 6.97，全树新最优，超s2-UNeXt 0.9166/8.38）。
- 追查线索（非结论）：s1的best_j分化0.29/0.28/0.42 vs s0的0.54/0.19/0.27（s0-J0垄断倾向）；s1 val_HD95 5.76 vs s0 6.72（验证态已分叉，非纯test噪声）；gamma_cv s1 0.0079 vs s0 0.0041。待独立seed复跑（b3后）。
- ckpt keys含ds_head+frame_only ✅；anchor 22M ✅；vf_deep=True（ec71ccf ckpt推断兼容，旧ckpt可load）✅。

## 3. b3短筛（6ep/60，UNeXt系，8 FINISHED）：方向已³见，待胜者全预算

| ds-lr | sm0 | sm0.01 |
|---|---|---|
| 5e-5 | 0.7033 | 0.7037 |
| 1e-4 | 0.7186/0.7006（双行，同值复现差1.8pt，短预算方差） | 0.7004/0.6977 |
| 2e-4 | **0.7277**（最高） | 0.7045 |

- 读数：ds-lr 2e-4最高（+2.4pt over 5e-5）；smooth 0.01全行低于同ds-lr的sm0（−0.3~−2.3pt，毒药方向延续）。但短预算方差带~2pt（1e-4双行差1.8pt），**不定最优，只定"2e-4/sm0可跑"**。
- 队列：4 task仍pending（对方卡3 + C3-DPFR-fair 1），C2/C3状态见§4。

## 4. C2/C3 fair对照：阻塞（容器侧缺upstream仓）

- baseline_v10.md草稿§19已记：upstream_BanditPM是独立git仓（未进DIAG仓），容器缺失→C2/C3 task失败。本地`upstream_BanditPM/`是untracked对照（git status `??`），tarball同步（sync.sh）只包DIAG-code/src/pyproject——C2/C3需tarball加包或容器内clone后重排 ⚠️。
- DPFR-fair既有数test 0.9348（006引）维持引用，不并表（协议未对齐，009 §7.2规则）。

## 5. R0-7应用口径：冻结largest单开，min_size/阈值待补

- post6主项结论延续：largest单开−2.65≈全开−2.78 ✅冻结；min_size {8,16,32,64} + val择阈未跑（val50本地未拉，阻塞延续）。

## 6. 与论文/H结论的对应关系

- v1.0锁定表以baseline_v10.md草稿为准，修正一条：R0-6 s0不进表（FLAKY候选），a1代表值=s1（0.9224/0.9185/6.97）。
- 后继分母：UNeXt系s2（0.9166/8.38）、DINO系R0-6 s1（0.9185/6.97）、FO分母DINO-FO 0.8880。
- 不产生Table对照数值（锁定表是起跑线，报方向不报点）。
