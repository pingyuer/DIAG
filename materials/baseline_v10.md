# Baseline v1.0 锁定表（R0复现批，裸口径）

split md5 `682f3d89`（test50/n=500）。ckpt md5见下（`md5sum outputs/*.pt`）。

| # | 配置 | ckpt (md5前8) | test Dice裸 | HD95 cdist/mirror | 双seed差 | 状态 |
|---|---|---|---|---|---|---|
| R0-1 | UNeXt-FO s0 | frameonly(旧世代) | 0.8078(既有) | 40.30/— | — | 沿用，s1待补 |
| R0-2 | UNeXt grid1 s0 | grid1_s0 `43214765` | 0.9110 | 11.00/12.76 | Dice 0.06pt | ✅锁定 |
| R0-2 | UNeXt grid1 s1 | grid1_s1 `a6cefc93` | 0.9116 | 10.54/11.62 | — | ✅锁定 |
| R0-3 | UNeXt s2 s0 | s2full_s0 `bca91243` | 0.9167 | 9.39/10.62 | Dice 0.27pt | ✅锁定 |
| R0-3 | UNeXt s2 s1 | s2full_s1 `e3112f01` | 0.9140 | 11.14/12.46 | — | ✅锁定 |
| R0-4 | DINO-FO s0 | dino_fo `29ed5a73` | 0.8880 | 9.74/10.57 | — | ✅分母落定(+9.0pt vs UNeXt-FO) |
| R0-5 | DINO-full s0 | dino_s_full `047666f1` | 0.9082 | 7.32/7.70 | Dice 0.03pt | ✅幂等锁定 |
| R0-5 | DINO-full s1 | dino_s1 `6b356080` | 0.9085 | 7.65/8.14 | — | ✅锁定 |
| R0-6 | DINO×ce0.05/rec1.0 s0 | r06_s0 `bcfca864` (val 0.9175/ep27) | 0.9066 | 7.66/8.18 | Dice 1.19pt | ⚠️seed内差超1pt，待追查 |
| R0-6 | DINO×ce0.05/rec1.0 s1 | r06_s1 `daeea123` (val 0.9224/ep28) | 0.9185 | 6.97/7.36 | — | ✅新最优点(val/test双高) |
| R0-7 | 应用口径 | largest单开≈全开(−2.65 vs −2.78, post6) | — | — | — | ✅冻结(min_size/阈值待补) |
| 006遗留 | eval vf-deep自适应 | ckpt推断`flow_f.vf.net.4/6`键→PCLF(deep) (ec71ccf) | — | — | — | ✅R0-6旧ckpt兼容 |
| C2/C3 | fair对照 | upstream_BanditPM为独立git仓(BanditPM源, 未进DIAG仓)；容器缺失→C2/C3 task失败 | 待 | 待 | 待 | ⚠️需tarball同步或容器内clone后重排 |

判定：R0-1~R0-5同ckpt/双seed内差<1pt ✅（最大0.27pt）；R0-6 s0/s1差1.19pt超阈→s0标FLAKY候选，不进锁定表。
R0-6 s1 val 0.9224/test 0.9185/HD95 6.97为全树新最优点（超s2-UNeXt 0.9166/8.38，超DINO基线0.9082/7.32）；s0 test 0.9066与s1差1.19pt超R0阈值1pt——标⚠️FLAKY候选（待独立seed复跑追查，见b3后）。v1.0以s1为a1代表值，s0不进锁定表。
