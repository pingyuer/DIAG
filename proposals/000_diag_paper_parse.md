# DIAG 论文解析：方法 / H 假设 / 表 1-2 口径梳理

依据：`proposals/35002_DIAG_Dynamics_Induced_Af.pdf` 全文（正文 §1–§5）。
性质：只解析、不设计。后续实验设计、代码、运行均不在本文档范围内。
幂等核对（2026-09-30）：`proposals/` 下除本 PDF 外无历史提议文档；`materials/` 为空；无重复提议。

## 1. 三分支方法结构

核心建模主张（§1, §3）：不把"解剖内容是什么"与"如何随时间变化"编码进同一个共享时空状态，
而是角色分离——目标帧做内容锚（content anchor）$x_t$，时序上下文参数化内容上的可组合群变换
$g_t \in G$，群作用 $\rho(g_t)x_t$ 给出时序条件化表示。跨区间变换经群运算复合（Eq.3）。

形式化（Eq.2）：

- $x_t = E_x(I_t)$：目标帧内容；
- $z_t = P_{\Delta t}(z_{t-1}, O(x_t))$：变换生成状态（历史演化 + 当前观测校正）；
- $g_t = \Gamma(z_t)$，$\hat{Y}_t = D(\rho(g_t)x_t)$。

可组合性实例化（Eq.4）：$\rho(g_t) = D_{\gamma_t} = \mathrm{Diag}(\gamma_t)$，$\gamma_t > 0$；
$G_{\mathrm{diag}} = \{D_a \mid a \in \mathbb{R}^n_+\}$，满足闭包、单位元、逆元与区间复合
$R_{t \leftarrow s} = D_{\gamma_t}D_{\gamma_s}^{-1} = D_{\gamma_t \oslash \gamma_s}$，
$R_{u \leftarrow s} = R_{u \leftarrow t}R_{t \leftarrow s}$。

### 1.1 Single-Frame Content Anchoring（§3）

- 目标帧 $I_t$ 经预训练图像编码器抽取细尺度特征 $F_{tf}$ 与下采样粗尺度特征
  $F_{tc} = \mathrm{Down}(F_{tf})$，$x_t = (F_{tf}, F_{tc})$（Eq.5）。
- 细尺度特征走向内容路径（特征门控 + mask decoder）；双尺度特征同时送入 PCLF 的观测编码器。
- 同一目标帧既提供直接分割内容，又提供校正动力学状态的观测。

### 1.2 Predict–Correct Latent Flow / PCLF（§3）

输入：当前双尺度特征 + 前一状态；经观测编码 → 连续预测 → 状态校正，输出当前隐动力学 $z_t$。

- Dual-scale states：维持粗状态 $S_{tc}$ 与细状态 $S_{tf}$；尺度 $q \in \{c,f\}$ 上
  $O_{tq} = O^q(F_{tq})$（Eq.6）；序列初始化 $S^q_1 = O^q_1$；$z_t = (S_{tc}, S_{tf})$。
  粗状态概括大范围演化，细状态保留局部变化。
- Continuous prediction：卷积动力学函数 $f^q_\phi$ 参数化连续时间向量场，显式 Euler 单步
  $\bar{S}^q_t = S^q_{t-1} + \Delta t\, f^q_\phi(S^q_{t-1})$（Eq.7）。
  预测态只依赖历史与真实时间间隔，是当前观测到达前的动力学先验；显式 $\Delta t$
  使同一向量场处理不同采样间隔。
- Observation correction：自适应校正门
  $K^q_t = \sigma C^q[\bar{S}^q_t, O^q_t]$（Eq.8），
  $S^q_t = (1-K^q_t)\odot \bar{S}^q_t + K^q_t \odot O^q_t$（Eq.9）。
  门在每个通道/位置调节历史预测与当前证据；校正后状态构成 $z_t$ 并送入 HDC 三分支。

### 1.3 Hierarchical Dynamic Conditioning / HDC（§3）

映射（Eq.10）：$\theta_t = H(z_t) = (\gamma_t, \beta_t, D_{tc}, D_{tf}, T_{tc}, T_{tf})$，
分别条件化图像特征、dense-prompt 路径、mask-query 路径。

- A. Feature-level affine gating（Eq.11–12）：细状态的轻量映射生成逐通道/逐位置
  scale 与 bias，$\gamma_t = \sigma G_\gamma(S^f_t)$（恒正，对应 Eq.4 对角群元），
  $\beta_t = G_\beta(S^f_t)$；$F^e_t = \gamma_t \odot F^{\mathrm{base}}_t + \beta_t$。
  $\gamma_t$ 实现内容上的乘性可组合变换，$\beta_t$ 做加性补偿。
  $F^{\mathrm{base}}_t$ 含辅助采样残差（实现细节，§3 Prediction and Optimization）。
- B. Dense spatial prompt（Eq.13）：双尺度 dense head 生成空间 prompt
  $D_{tc}, D_{tf}$，对齐分辨率后 $P_t = P_0 + \mathrm{Up}(D_{tc}) + D_{tf}$，
  叠加到预训练 prompt encoder 的 base prompt $P_0$。粗分支给全局区域上下文，细分支给局部细节。
- C. Region-level semantic tokens（Eq.14–15）：每尺度 $K$ 个可学习 region query 对展平隐状态做
  注意力，$T_{tq} = \Pi^q(A^q_t V^q(S_{tq}))$；粗细 token 门控并入预训练 mask query
  $Q^e_t = Q_0 + \alpha_c T_{tc} + \alpha_f T_{tf}$，$\alpha_c, \alpha_f \in (0,1)$。
  三分支联合产出 $F^e_t, P_t, Q^e_t$ 送 mask decoder。

### 1.4 Prediction and Optimization（§3）

- 预训练 mask decoder 预测候选 mask 及其质量
  $\{M^{(j)}_t, u^{(j)}_t\}_{j=1}^J = D(F^e_t, P_t, Q^e_t)$（Eq.16）；推理选最高质量候选。
- 有标注帧集合 $\Omega$ 上优化 CE + Dice，另加状态重建、位移平滑、下一帧特征预测、
  mask 质量监督（Eq.17，$\lambda_{\mathrm{rec}}, \lambda_{\mathrm{smooth}},
  \lambda_{\mathrm{flow}}, \lambda_{\mathrm{iou}}$ 权重；具体值在补充材料，正文未给）。
- 实现另保留 feature-sampling residual、stopped-gradient 短程 mask-token recurrence、
  identity-initialized boundary refinement（稳定训练/推理用；动态变换仍由上述群作用定义）。
- 下一帧只做辅助训练监督，推理保持在线因果。

## 2. H1–H4 假设与证据链

§4 Experimental Setup 明示评估围绕四个假设展开：

| 假设 | 内容 | 正文指定证据 |
|---|---|---|
| H1 target-frame content anchoring | 目标帧内容锚定分割，时序动力学只做调制 | frame-only 消融（去整条时序路径 Dice −2.63pt）；prompt-free 跨域结果（§4 Verifying） |
| H2 evolving predict–correct states | 时序状态持续演化：连续预测 + 观测校正 | 采样压力测试（Fig.5，5帧保留 ≥98.9%，3帧/70%随机缺失/40%时间戳抖动下 CAMUS 保留 95.9%/97.0%/97.5%）；相干性增益（Fig.4）；潜轨迹 R²=.983、状态干预、预测–校正统计（Fig.3a,b，§4 Latent-State Use） |
| H3 complementary hierarchical readouts | 同一动力学状态的三层读出互补 | Table 2 单项移除：去 spatial −0.78、去 affine −0.46、去 region −0.17；三者量级不等且全时序移除损失远大于任一单项 |
| H4 temporally ordered gain composition | 乘性增益构成时序有序、可组合变换 | Fig.3c,d：$\gamma$ 相邻步 CV ~1e−6 量级（$\beta$ 无同等正则性）；有序复合误差 .0063，时序 shuffling 后 .0134（2.13×）；正对角乘法的闭包/可逆作理论支撑 |

注意：Table 2 每行是同一全配置下的单分量移除，差值解释为相对贡献而非可加效应（正文明确声明）。

## 3. 表 1 主结果口径（§4 Comparison）

- 数据集与测试规模：CAMUS / EchoNet-Adult / EchoNet-Pediatric / CardiacUDA，
  测试集患者数分别为 50 / 1276 / 353 / 29。全部 patient-wise 划分，默认每视频采样十帧。
  Adult、Pediatric、CardiacUDA 只在有有效标注的帧上打分。
- 推理条件：DIAG 测试时不用 GT mask 也不用 box prompt（全自动块）。
- 指标：Dice + 统一 surface-based 实现的 HD95；时序行为用 centroid drift、area roughness、
  inter-frame Dice variation 表征。
- 选 checkpoint 规则：多调优 run 并存时用 validation Dice 选出完整 checkpoint，
  同一表行所有指标来自该 checkpoint。
- mDice：四数据集非加权宏平均，只报告覆盖全部四个数据集的方法。
- 上块全自动协议内 bold/underline 表最优/次优；下块 prompted 方法（首帧 GT mask † 或 GT box ‡）
  只作参考，不参与自动排名；"–" 表同协议下无可信结果。

数值（自动块，Dice↑ / HD95↓；mDice↑）：

| 方法 | CAMUS | Adult | Pediatric | CardiacUDA | mDice |
|---|---|---|---|---|---|
| 3D U-Net | .9162 / 5.46 | .8468 / 6.16 | .8436 / 7.09 | .7787 / 6.29 | .8516 |
| ConvLSTM | .9314 / 3.00 | .8475 / 6.48 | .8382 / 7.29 | .7518 / 10.17 | .8422 |
| GDKVM | .9368 / 6.13 | .9150 / 8.27 | .9052 / 4.80 | .7279 / 7.38 | .8712 |
| MemSAM | .9284 / – | .8976 / 8.33 | .8887 / 5.37 | – / – | – |
| OSA | .9284 / 5.83 | .7740 / 10.72 | .8186 / 4.17 | .5948 / – | .7790 |
| SAM2-memory | .9212 / 6.79 | .8915 / 7.63 | .8823 / 7.34 | – / – | – |
| DIAG | .9360 / 5.43 | .9146 / 5.75 | .9045 / 4.25 | .8092 / 6.18 | .8911 |

论文宣称要点（复现对照用）：CAMUS/Adult/Pediatric 上 DIAG Dice 第二，距最优仅
0.08 / 0.04 / 0.07 个百分点；CardiacUDA 自动 Dice 最优（.8092，超次优 3.05pt）；
全覆盖自动方法中 mDice 最高（.8911，超 GDKVM 1.99pt）；相对 GDKVM，Adult HD95
8.27→5.75（−30.5%）、Pediatric 4.80→4.25（−11.5%），Dice 差 <0.001；
CAMUS 上 ConvLSTM HD95 最低（3.00），DIAG 为 .9360 / 5.43（Dice 与 HD95 均第二）。

## 4. 表 2 消融口径（§4 Verifying）

- 数据集：CAMUS。ED/ES 报告端点 Dice，Mean 为十采样帧平均。
- 每行从同一全配置移除一个分量（Flow=连续传播，MS=双尺度，Aff.=仿射门控，
  Spa.=空间 prompt，Reg.=区域条件）。
- 全配置 DIAG：ED .9427 / ES .9218 / Mean .9317。

| 变体 | Flow | MS | Aff. | Spa. | Reg. | ED↑ | ES↑ | Mean↑ | Mean Δ |
|---|---|---|---|---|---|---|---|---|---|
| Frame-only | – | – | – | – | – | .9121 | .8987 | .9054 | −2.63pt |
| w/o spatial | ✓ | ✓ | ✓ | – | ✓ | .9123 | .9355 | .9239 | −0.78pt |
| w/o affine | ✓ | ✓ | – | ✓ | ✓ | .9386 | .9143 | .9271 | −0.46pt |
| w/o flow | ✓ | ✓ | ✓ | ✓ | ✓ | .9376 | .9165 | .9278 | −0.39pt |
| Single-scale | ✓ | – | ✓ | ✓ | ✓ | .9350 | .9246 | .9298 | −0.19pt |
| w/o region | ✓ | ✓ | ✓ | ✓ | – | .9407 | .9178 | .9300 | −0.17pt |

注：正文"removing continuous propagation or dual-scale states costs 0.39 and 0.19"
即 w/o flow −0.39、Single-scale −0.19，与上表一致。

## 5. 数据集 / 采样 / seed 与指标口径

- 数据集：CAMUS、EchoNet-Adult、EchoNet-Pediatric、CardiacUDA（测试患者数见 §3）。
- 采样：默认每视频十帧；Fig.5 压力测试：7帧/5帧、3帧、30%随机缺失、连续空洞、40%时间戳抖动。
- seed/划分细节：正文仅声明 patient-wise 划分与默认十帧采样，**未给出随机 seed、具体帧索引规则、
  训练/验证/测试划分数**；架构、训练、指标细节指向补充材料。正文引用补充材料的配置与优化细节
  在本次 PDF 中缺失——后续提议若涉复现，必须先索取补充材料或在 DIAG-code 中核对实际采样/seed 实现，
  不可自行编造。
- 指标口径：
  - 重叠：Dice（帧级；病例级 patient-level Dice 用于 Fig.4 增益：Adult +1.25pt、Pediatric +1.16pt）。
  - 边界：HD95（统一 surface-based 实现；具体实现细节在补充材料）。
  - 时序：centroid drift（Adult −41.7%、Pediatric −32.4%）、area roughness、inter-frame Dice variation；
    CAMUS 上 frame-wise 推理相对 DIAG 的四项时序误差比：Dice variation 7.9×、centroid jitter 2.3×、
    area roughness 12.5×、accumulated drift 3.8×。
  - 状态/变换诊断：潜轨迹 PC1–LV area 同步、心动相位线性解码 R²=.983（n=50）；
    连续预测使相邻状态平均变化 43.29（直接观测）→17.97（动力学先验），校正后 18.59；
    状态干预（cross-time shuffling / 噪声注入，n=25）Dice 降至 .9127 / .8676；
    $\gamma$ 相邻比率 CV ~1e−6 量级；有序 vs shuffle 复合误差 .0063 vs .0134（450 comparisons，95% CI）。

## 6. 后续实验提议可直接引用的锚点

- 复现对照基线 = Table 1 DIAG 行 + mDice .8911 + 上述 HD95；消融对照 = Table 2 全配置与六行 Δ。
- H1 验收：frame-only 缺口 ~2.63pt；H3 验收：三项移除 −0.78/−0.46/−0.17 且有序关系成立；
  H2 验收：Fig.5 保留率 + Fig.4 相干增益；H4 验收：CV 量级 + 有序/shuffle 2.13×。
- 缺失项（复现前必须补齐）：补充材料中的架构/训练超参、$\lambda$ 权重、HD95 统一实现、
  十帧采样与 seed、ED/ES 定义、patient-level 聚合方式。以 DIAG-code 实际实现为准，
  与本文档冲突时以代码为准并记录偏差。

## 7. 实现对照：论文要求 vs 当前代码（2026-09-30 实测）

实测：`DIAG-code/` 为空目录；`src/diag/` 仅 `__init__.py`（`main()` 打印 Hello）；
`pyproject.toml` 依赖只有 mlflow/numpy/Pillow/torch（CPU 源）；`materials/` 为空；
补充材料缺失（架构/训练超参、$\lambda$ 权重、HD95 实现、采样/seed 均无处可对）。

| 论文要求 | 当前实现 | 缺口 |
|---|---|---|
| Single-Frame Content Anchoring：预训练图像编码器 + $F_{tf}/F_{tc}$ 双尺度（Eq.5） | 无 | 编码器选型、Down 实现、$F^{\mathrm{base}}$ 残差均缺失 |
| PCLF：观测编码 $O^q$、Euler 预测 $f^q_\phi$ + $\Delta t$、校正门 $C^q$（Eq.6–9），$S^q_1 = O^q_1$ | 无 | 整个 §1.2 缺失；$\Delta t$ 显式间隔处理是 H2 复现关键 |
| HDC 三分支：$\gamma_t > 0$ 门控（Eq.11–12）、dense prompt（Eq.13）、region token + $\alpha_{c,f} \in (0,1)$（Eq.14–15） | 无 | 整个 §1.3 缺失；$\gamma$ 恒正约束（sigmoid）与 H4 诊断直接相关 |
| Mask decoder + 候选质量选择（Eq.16），在线因果推理 | 无 | 预训练 segmenter 基座未定 |
| 训练损失 Eq.17（CE+Dice+rec+smooth+flow+iou）及 $\lambda$ 权重 | 无 | 权重在补充材料中亦缺失，需 human 补给 |
| 评估：四数据集、十帧采样、Dice/HD95/时序指标、Table 1–2 对照 | 无数据、无指标代码 | 数据集获取路径、采样/seed、HD95 统一实现全部待定 |
| Fig.3–6 诊断（R²、CV、复合误差、保留率） | 无 | 诊断脚本需随实现一并补 |

结论：当前为零实现状态，论文 §3 的 Eq.5–17 无一落地。建议后续提议按
Anchoring → PCLF → HDC → 损失/decoder → 指标/诊断的顺序逐个下发 implementation 节点，
每个节点以本 §1–§5 对应小节为验收依据。
