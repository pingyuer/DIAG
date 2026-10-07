# 指标/诊断验收（eval n-20260930-144207-5d1 ← impl n-20260930-141014-6ff, commit 0d9a9cc）

口径依据：`proposals/000_diag_paper_parse.md` §3–§5（表1 Dice/HD95/mDice、表2 CAMUS ED/ES/Mean、时序与诊断指标定义）+ `proposals/001_diag_innovations.md` §3 指标/诊断清单项。

## 1. 结论：通过（带四条如实记录，不打回）

- 恒等测试、CAMUS 3 病人全链路、诊断协议链路、远端 CUDA——父节点断言基本属实，本地独立重跑逐项复现；发现两处需修正/澄清的记录见下（A、B），另两条为已知约束（C、D）。
- 记录 A（R2 历史 bug 已修复）：exp96 内 `smoke-metrics-camus3` 有三个 run。前两个（c294c598/a455e73a，同 sha=0402cd2c）`mean_r2_untrained=0.0`；最新（5f55f951，sha=767b5ad3=当前 HEAD）`mean_r2_untrained=0.918013`。本地重算三病人 R2=0.908/0.952/0.894，均值 0.918，与最新 run 一致——前两个 run 的 R2=0.0 是旧版 bug，验收以最新 run 为准。
- 记录 B（恒等测试表述修正）：anchor 称"static drift/rough/var=0, 1px-shift HD95=1.0"。独立重跑确认：**重复同一帧**的序列 drift/rough/var=0.0；HD95 自比=0.0、1px 平移=1.0、3px 平移=3.0（距离标度正确）；空 mask 时 HD95=inf（调用方过滤约定）。但注意"static"须理解为逐帧完全相同的序列——随机静态噪声序列的 var 可达 ~0.51（期望行为，非缺陷）。另 `_boundary` 用 4-连通腐蚀残差，2D 直接调用会得空边界（inf）；正确调用是 (…,H,W) batch 形式，smoke 内用法正确。
- 记录 C（R2 数值性质）：三病人 R2 均 ~0.9（均值 0.918）是在**随机初始化特征**上对 LV 面积的最小二乘拟合，属高维随机特征的伪相关，不代表生理跟踪。impl anchor 已诚实标注"R2~0.9随机特征相关性待训练后验"。**禁止**将 0.918 引用为论文 R²=.983 的对照值。
- 记录 D（干预协议为代理实现）：`state_intervention` 当前实现是对输入帧加噪（noise_std×0.05）而非真正的 cross-time state shuffle；docstring 已明示 shuffle 须由调用方 hook 在 model_fn 内施加。真正的论文式干预（shuffle→.9127 / noise→.8676）待训练后补 caller hook，本节点只验链路可跑。

## 2. 实测复核（本地 .venv torch 2.14 cpu）

| 断言 | 实测 |
|---|---|
| Dice 空约定（双空=1/单空=0/恒等=1/logits 输入） | ✅ 1.0/0.0/1.0/1.0 |
| HD95 自比 0 / 1px=1.0 / 3px=3.0 / 空=inf | ✅ |
| 重复帧 drift/rough/var=0 | ✅ 0.0/0.0/0.0 |
| patient 平均（3+1 帧→0.5） | ✅ |
| 合成 R2（线性信号→1.0） | ✅ 1.0 |
| 合成 γ CV（1e-6 抖动→~1e-6） | ✅ 1.43e-6（诊断链路可分辨论文量级） |
| 纯 gain 有序/shuffle（可交换性 sanity：双误差 0） | ✅ ordered=shuffled=0, ratio=0——证实对角 gain 算术可交换，与 impl anchor 的"须经特征做"注记一致 |
| 恒等模型采样压力保留率全 1.0 | ✅ 链路可跑 |
| CAMUS3 重算 dice=0.1043/0.0492/0.0221，均值 0.0586 | ✅ 与 run 指标 0.058564 一致 |
| `git diff 0d9a9cc HEAD -- 三文件` | ✅ 空，无漂移 |
| `_edt` 死代码 | ✅ grep 无残留（scipy/distance_transform 均无） |
| 远端 :32237 CUDA（HD95=1.0, Dice=1） | ✅ |
| `outputs/camus3` gitignored（产物不进 git） | ✅ |

## 3. 设计核验（相对论文口径的偏差，代码内均已声明）

- HD95：边界点集 cdist 精确实现；docstring PLACEHOLDER 明示统一 surface 实现待补充材料，数值若与论文实现细节不一致则记偏差。当前 CAMUS3 的 HD95 未进 mlflow（仅 dice/r2 打标），训练后对照表 1 时须补 HD95 上报。
- 时序三件套：drift 为累积 ‖c_t−c_1‖（nansum 跳空帧）；roughness 用二阶差分（离散加速度，防 flicker 漏检）；variation 为 GT-free 帧间 Dice——三者语义与 000 文档 §5 一致。
- 聚合：`patient_average` 先病人内平均再跨病人平均（防长 clip 主导），与论文 patient-level 口径一致。
- `phase_r2` 要求 areas 为 (T,B) 形状（1D 会 reshape 失败）；smoke 内传入 (T,B)，用法正确。属 API 约束，记于此。

## 4. mlflow run（exp96）

- `smoke-metrics-camus3` ×3（见记录 A），`tags.node=metrics-diag, source.name=DIAG-code/smoke_metrics.py`。
- 验收 run=5f55f951：`mean_dice_untrained=0.058564, mean_r2_untrained=0.918013`，code_sha=767b5ad3（当前 HEAD，无漂移）。

## 5. 与论文结论的对应关系

- 本节点是**指标管道存在性验证**（plumbing only），不产生 Table 1–2 对照数值：mean dice 0.0586 为未训练随机网输出；R2 0.918 为伪相关；保留率/干预/CV 均只验链路。
- 至此 Eq.5–17 + 指标/诊断全链路在结构层面闭环；H1–H4 实证验收的前提（CAMUS 训练、λ 真值、预训练基座、真实 dt、HD95 统一实现）仍缺，待训练节点。缺失项总表见各验收文件 §5/约束小节。
