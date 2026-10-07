# 对照表：论文引用数 + 舱内同协议行（010 §2.3格式）

规则（009 §7/010 §2.3）：舱外数一律标"论文引用数（协议未对齐）"，禁与舱内裸口径并表；
并表只允许 C1–C4 + R0-6 同协议行。

## A. 论文引用数（协议未对齐，不跑）

| 方法 | 出处 | CAMUS Dice/HD95 | 定位 |
|---|---|---|---|
| 3D U-Net | 论文基线 | .9162 / 5.46 | 配准派参照 |
| ConvLSTM | 论文基线 | .9314 / 3.00 | 时序一致性线（CAMUS HD95最低） |
| GDKVM | ICCV 2025（Wang et al.）| .9368 / 6.13 | Eq.1式共享状态实例，我方立论反面 |
| MemSAM | CVPR 2024（Deng et al.）| .9284 / – | SAM+时空记忆，prompted方法参照 |
| OSA | arXiv:2603.26188 | .9284 / 5.83 | 流形约束派，"弱群 vs 强流形"对照 |
| SAM2-memory | 论文引用 | .9212 / 6.79 | 记忆派参照 |
| DIAG（论文行） | 论文 | .9360 / 5.43 | 复现目标 |

## B. 舱内同协议行（split md5 `682f3d89`，裸口径，test50）

| 行 | 配置 | test Dice | HD95 cdist/mirror | 来源 |
|---|---|---|---|---|
| C1 | UNeXt-FO | 0.8078 | 40.30 / — | R0-1既有 |
| C1 | DINO-FO | 0.8880 | 9.74 / 10.57 | R0-4 |
| DIAG | UNeXt s2（ce0.05/rec1.0）| 0.9167/0.9140 | 9.39/10.62 / 11.14/12.46 | R0-3双seed |
| DIAG | DINO×ce0.05/rec1.0 s1 | 0.9185 | 6.97 / 7.36 | R0-6 s1（a1代表值；s0 FLAKY不进表） |
| C2 | GDKVM fair s0 | 0.9337 | 待（upstream无HD95上报，只有Dice/IoU） | exp98 `3049b212` (val 0.9331) |
| C2 | GDKVM fair s1 | 0.9318 | 待 | exp98 `96f4aecb` (val 0.9330)，双seed差0.19pt |
| C3 | DPFR-fair s0 (prompt-free) | 0.9316 | 待（upstream无HD95上报；ckpt回传重算HD95为后继项） | exp99 dpfr-fair-s0 (val 0.9285) |
| C3 | DPFR-fair s1 (prompt-free) | 0.9332 | 待 | exp99 dpfr-fair-s1 (val 0.9268)，双seed差0.16pt |
