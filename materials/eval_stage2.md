# stage-2 联合验收（eval n-20261004-065301-abc ← impl n-20261003-092800-f75）

口径依据：006 stage-2（ce{0.01,0.05}×rec{0.5,1.0}×seed{0,1} 短筛 + 胜者全预算双 seed；P9 联合效应）。

## 1. 结论：联合成立但幅度小（+0.5pt，双 seed 同向；K 0.97 是红灯不是勋章）

- 短筛 8/8 FINISHED（6ep/60，ce×rec×seed 全）：0.68–0.72 带，排序 ce0.01-rec1.0（0.71/0.70）微顶；
  seed 内差 0.6–2.9pt，短预算方差带内——短筛只定"ce0.05-rec1.0 可跑"，不定最优。
- 全预算双 FINISHED（ce=0.05/rec=1.0，400/30ep）：s0 0.9163 / s1 0.9175（Δ=0.12pt）。
  相对格 1 ON 均值 0.9110：**+0.5pt，双 seed 同向** ✅（s0 +0.43 / s1 +0.75）。
- test（s1 ckpt，本地重跑）：frame-mean **0.9166** / HD95 8.38 / mirror 9.09 /
  ES-proxy 0.9074 / small-cavity 0.9009（n=14）/ boundary-band 0.6541 / first 0.885 / last 0.909。
  相对 ON s0 test 0.9136/8.22：+0.3pt，同分布上沿。

## 2. K 0.97（记录：增益伴随门控恶化，不是健康信号）

- s2-s1 K 终值 0.9718 / s0 0.9600 vs 格 1 s0 0.92 / s1 0.88——联合把 K 推到 O3 动作线上方（>0.95）。
  ce 降权（0.05）削弱像素校准监督，门向观测侧/IR 走更远；val 照涨说明观测直通在此预算下仍是上升方向。
- best_j 分化正常（0.43/0.28/0.29）；α 0.12 未动；gamma_sat=0；lr 调度触发双 run 一致。
- 判读：+0.5pt 的"联合增益"与"K 恶化"同源——P4（K 不贴边）在此配置下**更差**了。
  后继若追 K 回落，ce 不可再降（方向：ce 回调或 rec 探头加权）。

## 3. 实测复核

- params `l_ce=0.05/l_rec=1.0` 双 run ✅；tags `phase=grid2_detach_ablation` 系 runner 复用旧 phase 名（记一行，非本格重标）。
  `lambda` tag 仍占位（实现用 params 为准）。
- ckpt `outputs/s2full_s{0,1}_best.pt`（md5 bca91243…/e3112f01…，gitignored）：keys 含 ds_head ✅（epoch 27/29）。
- ds 1.08（双 run 终值 >1）——ds 在 ce 低监督下学出放大步长；方向待格 3 语境（已证伪时钟，此处只记数）。
- eval 侧 bug 修：`eval_test.py:146` 变量 `a` 覆盖 argparse（`a.mirror` 崩）→ `_a`，已 commit 待推；
  本次 test 用修后重跑（mirror 9.09 落盘）。

## 4. 与论文/H 结论的对应关系

- P9（关某项升则有害 + 联合方向）：ce0.05×rec1.0 联合 +0.5pt 双 seed 同向——方向先验转弱正向证据；
  单点禁引用延续（幅度 < seed 方差史，但本次是双 seed 同向，有别于单点）。
- 主干更新：s2（ce0.05/rec1.0）接替格 1 ON 为当前最优 UNeXt 主干（val 均值 0.9169 / test 0.9166）。
- 不产生 Table 对照数值（0.5pt 在跨 seed 带内，报方向不报点）。
