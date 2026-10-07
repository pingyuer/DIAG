# 008 相性搜索验收（eval n-20261002-130244-ed3 ← impl n-20261001-165210-bad）

口径依据：008 提案（frozen 只前向 + frame-only 对照；DINOv2-s/b + MedSAM + SAM 四候选）+ 正向线（P1：内容锚表达力单调决定 frame-only 上限）。

## 1. 结论：部分通过（P1 部分证实；C/D 未跑；单 seed，禁点估计）

- DINO-small 全链 30ep（`dino-s-full-30ep` FINISHED，seed0，400/30ep）：best val **0.9188**，val_hd95 5.6。
  相对 UNeXt grid1-s0（0.9120/9.1）：val **+0.7pt**，HD95 **−3.5**（val）/ test −1.5（7.34 vs 8.22，+后处理）。
- test（本地重跑，`--anchor dinov2-small`）：frame-mean **0.9081** / HD95 7.34 / mirror 7.69 / first 0.9126 / last 0.8904。
  相对 UNeXt s0 test（0.9136/8.22）：dice **−0.5pt**，HD95 −0.9。val-test gap −1.1pt（UNeXt s0 为 +0.2pt）。
- DINO frame-only 短预算（6ep/60子集/seed0）：small 0.8585 vs base 0.7861（+7.2pt，小胜大）。
- MedSAM / SAM（C/D）：**零 runs，未跑**（权重链路未通）。提案四候选只落地一半，记为未跑非证伪。

## 2. 预言判定（P1：内容锚表达力单调）

- ✅ 方向：frozen DINO-small 全链 val 超随机 UNeXt（+0.7pt），HD95 大幅收窄；frame-only 短预算 small>base。
- ⚠️ 保留：test dice 反超 −0.5pt + val-test gap −1.1pt（UNeXt 无此 gap）——"表达力单调"在 val 成立，在 test 不成立。
  单 seed 下不判胜负；需 seed1 DINO 全链复跑 + 双 test 对照才能转正。
- P1 状态：部分证实，待 seed1。

## 3. 联动诊断（ds/K/γ，DINO run 内）

| 信号 | DINO-s-full | UNeXt s0/s1 | 解读 |
|---|---|---|---|
| ds_mean 终值 | 0.67（min 0.55，真动且幅度大） | 0.92 / 1.01 | 好特征下 ds 学出形状（不再恒等）；方向/幅度仍待格 3 真间隔终验 |
| K 终值 | 0.83 | 0.92 / 0.88 | K 漂持续——换好特征也没拉回，VectorField 弱实锤+1（架构问题，非特征问题） |
| gamma_sat | 0 全程 | 同 | 健康 |
| gamma_cv | 0.0006→0.0076 | 0.003→0.010 | 同量级，未收敛（待格 4） |
| best_j 分布终值 | 0.37/0.22/0.41 | — | 三候选分化，无垄断 |

- first/last 翻转：DINO test first 0.9126 / last 0.8904 vs UNeXt s0 first 0.87 / last 0.91。
  DINO 首帧强、尾帧弱——单 ckpt 观察，不作结论（待 seed1 看是否复现）。

## 4. 实现核验

- `src/diag/dino_anchor.py`：frozen（`requires_grad_(False)` + `no_grad` 前向）✅；CLS 丢弃 + 18×18→128 上采样 + 1×1 适配至 C=96 ✅；
  PCLF/HDC/Decoder 零改动（接口冻结成立）。
- `--anchor` 开关：train + eval 双侧 ✅（本次 test 即 `--anchor dinov2-small` 实测）。
- 灰度→3ch + ImageNet 归一（echo stats≠ImageNet，偏差已声明）。
- ckpt `outputs/dino_s_full_best.pt`（96MB，md5 047666f1…，gitignored）：keys 含 `ds_head`+`frame_only=False` ✅；
  anchor 22M 参数（DINO-small frozen 21M + 适配层）。
- backbone tag：`dinov2-small-frozen-pretrained` ✅（UNeXt 行仍 `-random-init-DEVIATION`）。

## 5. 与论文/H 结论的对应关系

- 不产生 Table 1–2 对照数值（单 seed 短…全预算单 seed）。P1 部分证实只回答"预训练锚方向对"，
  H1 分母重标需 seed1 + MedSAM 到位后。
- MedSAM（真 P0/Q0，HDC-B/C 名副其实）仍是最大未探项；SAM 自然版对照（C赢D才证"医学"有用）同样未跑。
