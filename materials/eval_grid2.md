# 格 2 detach 消融验收（eval n-20261003-135313-7ca ← impl n-20261003-092749-c1f）

口径依据：006 §2格2（on/off 全预算双 seed；|Δ|<0.2pt 冻格；K/γ/α 联动）+ P7（detach 断捷径；关则 K 更漂）。

## 1. 结论：冻格（detach 无增益证据；P7 后半证伪）

| seed | ON（格1）best | OFF（格2）best | Δ(OFF−ON) |
|---|---|---|---|
| s0 | 0.9120 | 0.9101 | −0.19pt |
| s1 | 0.9100 | 0.9186 | +0.86pt |
| 均值 | 0.9110 | 0.9144 | +0.34pt |

- s0 负、s1 正，方向不一致；均值 +0.34pt 在 seed 方差带内——detach 开关无系统增益，冻格（flag 保留，不删）。
- test（OFF s1 ckpt，本地重跑）：0.9124 / HD95 8.78 / mirror 9.54 / first 0.876 / last 0.907——
  vs ON s0 test 0.9136/8.22：−0.12pt，同分布。
- K：OFF s0 0.92 / s1 0.84 vs ON s0 0.92 / s1 0.88——关 detach 后 K 照漂，无"更快漂向 1.0"；
  P7 后半（关则 K 更漂）证伪。α_c/f 全行 0.12 未动；best 三候选分化正常；gamma_sat=0。

## 2. 实测复核

- 双 FINISHED（0225309f / fad1b6e7），同 budget 400/30ep；params `no_detach_region_s=True` ✅；
  ckpt keys 含 ds_head ✅（epoch 26/29，sha 为 tarball 口径）。
- 开关实现：`--no-detach-region-s` 透传 RegionTokenHead（b7aa171，读验一致）。
- test patient-avg==frame-mean（等长恒等）；split_md5 682f3d89 ✅。
- ckpt 本地 `outputs/grid2_off_s{0,1}_best.pt`（md5 61f797ea…/ec96c5ac…，gitignored）。

## 3. 与论文/H 结论的对应关系

- P7 前半（单侧性）成立既有；后半（隔离增益）无证据——H3 region 路径贡献（Table 2 −0.17）在我方架构下未复现。
- 不产生 Table 对照数值（单消融行，|Δ| 在噪声带内）。
