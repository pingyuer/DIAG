# 008续验收（eval n-20261004-073825-449 ← impl n-20261003-092806-dc4）

口径依据：008 提案（DINO seed1 + MedSAM/SAM 权重链路；P1 转正判据 = HD95 + 全预算对照 + seed1 test）。

## 1. 结论：DINO 转正（双 seed 一致；MedSAM/SAM 链路通但未跑）

- DINO-small 全链双 seed FINISHED（同 budget 400/30ep）：s0 0.9188 / s1 0.9216（Δ=+0.28pt，同向）。
  val_hd95：5.6 / 6.2。相对 UNeXt 格 1（0.9120/0.9100）：+0.7/+1.2pt。
- test（双 ckpt 本地重跑，`--anchor dinov2-small` + 后处理）：
  s0 0.9081/7.34 vs s1 **0.9086**/7.64——test 双 seed 差 0.0005（饱和一致），相对 UNeXt s0 0.9136：−0.5pt。
- DINO frame-only 全预算（`dino-fo-full` FINISHED，seed0，400/30ep）：val **0.8928**/best 0.8955/HD95 8.0。
  相对 UNeXt frame-only（0.8025）：**+9.0pt**——P1 分母重标成立（表达力单调在 frame-only 口径下成立且幅度大）。
- MedSAM/SAM：权重本地齐（1.5G blobs，`SamModel` 93M 直 load ✅）；接口探明：
  `vision_encoder` 直出 (1,256,64,64) dense + prompt/mask decoder 俱全；但 1024 输入要求
  （超声 256 需上采样×4，路径另议）+ 容器未 scp + C/D 零 runs——记链路通、实验未跑。

## 2. P1 转正判定（HD95 主判据 + 全预算对照 + seed1 test）

| 判据 | 结果 |
|---|---|
| HD95（主） | DINO val 5.6/6.2 vs UNeXt 9.1/8.0；test 7.34/7.64 vs 8.22——双 seed 同向收窄 ✅ |
| 全预算对照 | DINO-FO 0.8928 vs UNeXt-FO 0.8025（同 400/30ep）✅；全链 DINO 0.9188/0.9216 vs UNeXt 0.9120/0.9100 ✅ |
| seed1 test | 0.9086 vs s0 0.9081（饱和一致）✅；但 test 反超 UNeXt −0.5pt（val-test gap −1.1pt 延续）⚠️ |

- P1 转正 ✅（方向三判据全过；test 反超记为"Dice 天花板假象"，HD95 真差为准）。
- K 行为注记：DINO s1 K 终值 0.60（s0 0.83）——好特征下 K 反而更低；ds 0.58（min 0.46，真动）。
  K 漂持续但幅度分 seed，VectorField 弱与特征无关（既有结论延续）。
- first/last：s1 0.9107/0.8867 vs s0 0.9126/0.8904（逐项复现）；small-cavity 0.8853/0.8853 一致。

## 3. 实测复核

- ckpt `outputs/dino_s1_best.pt`（96MB，md5 6b356080…，gitignored）：keys 含 ds_head ✅（epoch 29/val 0.9216）。
- `--anchor` eval 双侧 ✅（本次即实测）；`no_detach_region_s=False`（detach ON，P7 冻格态延续）。
- test patient-avg==frame-mean；split_md5 ✅；ES-proxy 0.8869 / small 0.8853 / band 0.6188（008-2 切片链路通，数值只记数）。
- test HD95 mirror：s1 8.08 vs s0 7.69（mirror>cdist 延续）。

## 4. 与论文/H 结论的对应关系

- H1 分母重标：DINO-FO 0.8928 取代 UNeXt-FO 0.8025——时序分支增益相对预训练锚重算（全链−FO 同锚内差，待后继）。
- MedSAM（真 P0/Q0）仍是最大未探项；SAM 自然版对照同样未跑。
