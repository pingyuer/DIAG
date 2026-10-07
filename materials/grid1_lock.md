# 格 1 主干锁定验收（eval n-20261001-174639-077 ← impl n-20261001-133842-d0a）

口径依据：006 格 1（全预算 30ep×400×2seed；停止三规则）+ 正向线（本格只锁 P3/P4/P5 基线，不追点数）。

## 1. 结论：通过（主干锁定 0.91；seed 方差终结；ds/K 联动待判）

- 双 FINISHED（s0 bb59e46b / s1 4c8b53a6），同 code f5e41cc8，同 budget：best val **0.9120 / 0.9100**（Δ=0.2pt）。
- 欠训区 8.6pt 方差 → 全预算 0.2pt：方差终结 ✅，006 格 1 目标达成。
- test（s0 ckpt，本地重跑）：frame-mean 0.9136 / HD95 8.22 / mirror 8.95（+后处理）/ first 0.87 / last 0.91。

## 2. 预言基线锁定（P3/P4/P5，本格真正产出）

| 预言 | s0 | s1 | 判定 |
|---|---|---|---|
| P3 ds 学步长 | ds 1.0→0.92（首次真动，不再恒等） | ds 1.0→1.01（反向微动） | ⚠️ 方向不定：同配置下 ds 可上可下，幅度 ~8% vs ~1%；"学步长"成立，"学对方向"未定——待格 3 真间隔终验 |
| P4 K 不贴边 | K 0.52→0.92（漂移未止） | K 0.50→0.88 | ❌ 两 seed 一致漂向观测：ds 动了也没拉回 K；VectorField 弱实锤（与 anchor 一致） |
| P5 γ 健康 | sat=0 全程；mean 0.881→0.869 | 同左 | ✅；CV 0.003→0.010（量级未收敛，待格 4） |

- lr 调度触发（3e-4→1e-4，两 run 一致）；gamma_cv 上升（0.003→0.010）非下降——CV 不是单调收敛量，格 4 需重看判据。
- 停止三规则：ep15 val>0.85 ✅；HD95<25 ✅；K 未越 0.95 线（0.92 逼近但未触发）。

## 3. test 三档（s0 ckpt，独立重跑复核）

frame-mean 0.9136 / patient-avg 0.9136（等长恒等）/ HD95 8.22 / mirror 8.95 / drift 53 / rough 152 / var 0.043。
相对 ds-full 世代（0.9119/8.29）：+0.17pt/−0.07，同分布（ckpt 含 ds_head 键后首次可复载，债务关闭）。

## 4. 产物与可追溯性

- `outputs/grid1_s0_best.pt`（39MB，md5 43214765…，gitignored）：keys 含 `ds_head`+`frame_only` ✅；epoch 28 / val 0.912 / sha f5e41cc8。
- s1 mlflow 仍有 RUNNING 僵尸行（018395d9/45d75ee1）+ FINISHED 行（4c8b53a6/8fa7f287）同值——脚本重跑残留，以 FINISHED 为准（与过往模式一致）。
- s1 ckpt 未 pull（远端 best.pt 同代可拉，后继 test-s1 时补）。

## 5. 记录修正

- anchor"双 seed 差 0.2pt"✅；"s1 ep2 假死乌龙"为 log 探针问题，非训练问题。
- anchor"K 漂 VectorField 弱实锤"：s1 复现同向漂移，支持该判；但"弱"需格 3（ds 真间隔）+ Obs/VectorField 范数比才能定量，当前为定性。
