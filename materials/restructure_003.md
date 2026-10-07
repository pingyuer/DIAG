# 003 重构验收（eval n-20260930-184143-970 ← impl n-20260930-182923-a26, commit 4edd581）

口径依据：`proposals/003_ds_detach_svf.md` §5 落地清单（1 ds 头 / 2 region 单侧 detach / 3 SVF flag-off / 4 ds 窗口列 / 5 不做三项）。

## 1. 结论：通过（带四条修正/约束记录，不打回；另有一处实现侧数学疑问记为观察项）

- ds 恒等延续、detach 单侧性、SVF flag-off 恒等、r003-smoke 32 指标 live——父节点断言基本属实，独立重跑逐项复现；修正见记录 A–D。
- 记录 A（detach 验证方法修正）：初次验证中旧 kernel 常驻内存，`rh.tokens()` 走到 stale code 得出"s.grad≠None"的假阳性；`importlib.reload` 后 region-path `s.grad=None` ✅，queries/k_proj 梯度仍在（单侧性成立）。结论：detach 实现正确，假阳性是验证环境问题（已排除），非代码缺陷。教训：本文件 detach 结论以 reload 后为准。
- 记录 B（grid 修正确认）：`identity_grid` 用 arange 中心（+0.5）+ `align_corners=False` 全链统一 ✅；OFF 恒等 warp maxdiff=0.0 ✅；零速 `exp(0)` 恒等 ✅。半像素 bug 修复属实。
- 记录 C（SVF 未接主链）：`SVFWarpHead` 无任何 import/调用点（`SVF_NOT_IMPORTED`）✅——符合"flag 默认关""转正标准写进 tag，否则删除"的提议要求。当前是孤立模块 + 恒等验证，无主链行为影响。`max_disp=0.05` 截断 + smooth 三件套读验齐；实际 ON 分支初值位移 ~1e-4（小方差 init 生效）。
- 记录 D（r003-smoke 一致性）：run `0366c557`（sha=a95852dc==本地 `.code_sha`==HEAD 4edd581 前身？见下）32 live 键：wstep 18（16+ds_mean/ds_min）+ epoch 旧 5 + wepoch 新 7 + diag 1 + best 1。`ds_mean=ds_min=1.0`（初值恒等延续 ✅，未塌零）；`watch/ds_min=0.05` tag 已写 ✅。`best_j1=1.0` 垄断（O4 观察态，2ep 早期）；`gamma_cv=2.8e-4` 未训练预期态。

## 2. 实测复核

| 断言 | 实测 |
|---|---|
| ds 初值 1.0（softplus⁻¹(1) bias≈0.54，恒等延续） | ✅ 随机/静态输入均为 1.0；`delta_scale=8` 解决 pooled 能量 O(0.05) 量级问题 |
| ds detach（tokens 无梯度）+ clamp≥1e-3 | ✅ `requires_grad=False`；floor 全员满足 |
| ds 可区分性（训练后） | ✅ metric 扰动后 still 1.0 vs moving 0.92 |
| 能量 \|d\|（E>0）vs 有向差分对称陷阱 | ✅ E>0；设计修正论证成立 |
| region 单侧 detach（S 无梯度 / queries 学） | ✅ reload 后 s.grad=None，queries/k_proj 梯度>0 |
| gate 路仍通 S | ✅（HDC 验收既有结论，不受 detach 影响——detach 只在 `RegionTokenHead.tokens` 内） |
| SVF OFF 恒等（phi/v/smooth/delta） | ✅ v=0/smooth=0/delta=0，warp maxdiff=0.0 |
| SVF ON 位移 ≤0.05 | ✅ 初值 ~1e-4（含 tanh×max_disp 截断） |
| ds 接线（ones→ds 输出，`ds_vec[0]`，B=1） | ✅ 代码读验；`dts` tag 已改为 `ds-head-learned-DEVIATION-fallback-ones` |
| ds 窗口列 + `watch/ds_min` tag | ✅ wstep/ds_mean/ds_min live；tag=0.05 |
| git sha | ⚠️ 本地 `.code_sha=a95852dc` ≠ `git log HEAD=4edd581`——`.code_sha` 是 tarball sha256（全链验收记录 C 已述），非 git object；run tag 与 `.code_sha` 文件一致即通过 |

## 3. 观察项（非打回，implementation 后继留意）

- SS6 逆一致性：`exp(v)∘exp(-v)` 在 v~N(0,0.02) 下 maxdiff≈1.6（归一化坐标，~13px@16px 图），v=0.001 时才降至 0.2。实现 `exp` 的 compose 步（`delta*2 - ident`）疑似位移加倍语义偏差。**但**：① SVF 未接主链，当前零行为影响；② ON 初值位移 ~1e-4（逆误差可忽略）；③ 转正标准（`flow_prompt_delta>0` + 诊断三项）本就会在启用前重验。故记为观察项，不打回；SVF 转正前必须重验逆一致性并修 compose 数学。

## 4. 与论文/H 结论的对应关系

- 003 是**强力重构**（提议自述"不与 Eq.5–17 强绑定"），本节点只验清单落地，不产生 Table 1–2 对照数值。
- ds 终验（变间隔/缺失采样 Fig.5 清单）待后继；SVF 转正/删除待 `flow_prompt_delta` 证据；原型池/scan/backbone 替换明确不做 ✅（`SVF_NOT_IMPORTED` 外无新增结构依赖）。
- 缺口延续：λ 真值、预训练基座、ED/ES 定义——均未在本节点解决，记于此。
