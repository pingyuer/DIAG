# DIAG 创新点梳理 + BanditPM 代码差距审阅

依据：`proposals/000_diag_paper_parse.md`（§1–§5 锚点）+ `upstream_BanditPM/` 代码树实测（README、`dpfr/model.py`、`dpfr/losses.py`、`dpfr/grid.py`、`configs/model/dpfr.yaml`、`model/modules/unext/official.py`、`dataset/frame_index.py`）。
只整理、不实现。幂等核对：`proposals/` 下仅 000 解析文档，无重复提议。

## 1. 论文创新点清单（4 条，每条标 H 归属与验收数）

### I1. 内容锚与变换角色分离（Eq.2–3）→ H1
- 目标帧做内容锚 $x_t$，时序上下文只参数化内容上的可组合群变换 $g_t \in G$，
  $\hat{Y}_t = D(\rho(g_t)x_t)$；反对一切塞进同一隐状态的 Eq.1 式设计。
- 验收：frame-only 消融 Mean −2.63pt（Table 2）；CardiacUDA prompt-free 最优 .8092。

### I2. PCLF 预测–校正隐流（Eq.6–9，显式 Δt）→ H2
- 双尺度状态 $z_t=(S_{tc},S_{tf})$，$S^q_1=O^q_1$；Euler 单步连续预测
  $\bar{S}^q_t = S^q_{t-1} + \Delta t\,f^q_\phi(S^q_{t-1})$（只依赖历史+真实间隔）；
  自适应门 $K^q_t$ 融合预测与当前观测（Eq.8–9）。在线因果。
- 验收：Fig.5 保留率（5 帧 ≥98.9%；3 帧/70%缺失/40%抖动下 95.9%/97.0%/97.5%）；
  相邻状态变化 43.29→17.97（预测）→18.59（校正）；相位解码 R²=.983；
  状态干预（shuffle/噪声）Dice →.9127/.8676。

### I3. HDC 三层读出（Eq.10–15，γ>0 对角群 Eq.4）→ H3/H4
- A. 特征仿射门控：$\gamma_t=\sigma G_\gamma(S^f_t)>0$（恒正=对角群元），
  $F^e_t=\gamma_t\odot F^{\mathrm{base}}_t+\beta_t$；
- B. dense 空间 prompt：$P_t=P_0+\mathrm{Up}(D_{tc})+D_{tf}$；
- C. region 语义 token：$Q^e_t=Q_0+\alpha_cT_{tc}+\alpha_fT_{tf}$，$\alpha\in(0,1)$。
- 同一 $z_t$ 读出三层，调制发生在 decoder **之前**（特征/prompt/query 级），无 logits 级融合。
- 验收：Table 2 单项移除 −0.78（spatial）/−0.46（affine）/−0.17（region），有序且互补；
  $\gamma$ 相邻比率 CV~1e−6（$\beta$ 无）；有序复合误差 .0063 vs shuffle .0134（2.13×）。

### I4. 在线因果推理 + Eq.17 训练 → H1/H2
- 推理：decoder 输出 J 候选 mask+质量分，选最高质量；测试无 GT mask、无 box；下一帧仅训练监督。
- 训练：$L=L_{CE}+L_{Dice}+\lambda_{rec}L_{rec}+\lambda_{smooth}L_{smooth}
  +\lambda_{flow}L_{flow}+\lambda_{iou}L_{iou}$（λ 在补充材料，正文缺失）；
  另有 feature-sampling residual、stopped-grad 短程 mask-token recurrence、
  identity-init boundary refinement。

## 2. 代码对照：upstream_BanditPM 相对 I1–I4

### 2.1 GDKVM 路径：已有（作对照基线，非 DIAG）
- KV memory + gated delta rule 的时空联合状态 = 论文 Eq.1 式"共享表征"的实例，
  正是 DIAG 立论要反对的一侧。**结论：GDKVM 路径保留为 Table 1 对照基线（GDKVM 行），
  不得当作 DIAG 实现起点。**

### 2.2 DPFR 路径 vs I1：部分形似，语义冲突
- 有：anchor（UNeXt decode）→ prompt 调制 → flow 细化 → residual fusion 的多阶段链条，
  与"锚定+调制"字面相似；`DPFRPromptModulator` FiLM 模式有 scale+shift。
- 缺/冲突：
  - DPFR 的"变换"是 **logits 级 warp**（`DPFRFlowHead` 输出 tanh 有界位移场，
    `grid_sample_logits` 双线性重采样）+ logits 级门控残差融合
    （`DPFRResidualFusion`：final = anchor + g1·(prompt−anchor) + g2·(flow−prompt)）。
    DIAG 无任何显式光流 warp、无 logits 级融合，变换只发生在特征/prompt/query 级。
  - FiLM 实现是 `feat * (1.0 + scale)`（`official.py:_apply_modulation`），scale 可正可负、
    无恒正约束；gated_add/add 模式更是纯 shift。**与 Eq.4/11 的 $\gamma_t=\sigma(\cdot)>0$
    对角群要求直接冲突**，H4 的 CV/复合诊断在该实现上无意义。

### 2.3 DPFR 路径 vs I2：核心冲突（同名异构之首）
- DPFR 自称 "Dual-Prompt **Flow** Refinement"，但其 flow = 上述 logits warp 头，
  **不是** PCLF 的隐动力学预测–校正。
- `DPFRDualPromptEncoder` 是 **全窗双向 transformer**（cls+多尺度图像token+mask token，
  `time_embed` 位置编码，4 层全注意力）：一次性看全 T 帧，非因果、无 $S_{t-1}\to S_t$ 递推、
  无 $f_\phi$ 向量场、无 Euler 步。
- **显式 Δt 缺失**：`dpfr/`、`dataset/` 中 grep 无 delta_t/timestamp/time_interval；
  时间只以 `time_embed[:, :time]` 顺序位置存在。I2 验收核心（变间隔/缺失/抖动鲁棒性，
  Eq.7 的 Δt）在此架构上无法表达。
- 状态初始化 $S^q_1=O^q_1$、校正门 $K^q_t$（Eq.8–9）在 DPFR 中无对应物
  （最接近的是融合门，但作用于 logits 残差而非隐状态）。

### 2.4 DPFR 路径 vs I3：逐条点名
| DIAG 要求 | DPFR 现状 |  verdict |
|---|---|---|
| A. $\gamma_t>0$ 逐通道/逐位置乘性门控（sigmoid） | FiLM `(1+scale)` 可负；常用 gated_add 纯 shift | 冲突，需重写 |
| B. dense prompt 叠加到预训练 $P_0$（Eq.13） | 有 prompt 调制注入 backbone low/mid/high/dec，但作用对象是自家 UNeXt，无 $P_0$ 接口 | 缺失（缺预训练 segmenter 基座） |
| C. region token 门控并入 $Q_0$（Eq.14–15） | 无 region query、无注意力 token、无 $Q_0$ | 缺失 |
| 调制点在 decoder 之前 | prompt 调制在 decode 内，flow/fusion 在 logits 后 | 部分冲突（flow/fusion 须删） |

### 2.5 DPFR 路径 vs I4：训练泄漏冲突 + 基座缺失
- **GT 泄漏**：`_build_mask_prompt` 训练时以 `gt_prob`（1.0→0.5 退火）把 GT mask
  直接作为 prompt 输入（teacher forcing）；DIAG 的校正观测 $O(x_t)$ 永远来自当前帧特征，
  测试 prompt-free。DPFR 的 GT 条件训练与 H1/H2 的 prompt-free 验收冲突，
  复现 DIAG 时必须关闭（`mask_prompt_train.use_gt: false` 或删该路径）。
- decoder 无 J 候选+质量选择（Eq.16）；损失是 final/anchor/prompt/flow_seg 四路 CE+Dice
  + flow mag/smooth/temp（`dpfr/losses.py`），与 Eq.17 的 rec/smooth/flow/iou 分解不对齐，
  λ 口径亦不同。
- 基座：DIAG 要求预训练图像编码器 + 预训练 mask decoder（带 $P_0/Q_0$ 接口）；
  DPFR 用 UNeXt（README 自述 local 疑似未预训练 "unpretrained UNeXt anchors"）。
  **预训练 segmenter 基座是当前最大外部依赖缺口。**

### 2.6 可复用的部分（明确给 imp）
- 数据管线：`dataset/`（echo/camus/cardiacuda/pediatric 注册表、`frame_index.py` ED/ES 解析、
  patient-wise 划分）+ `configs/data/*.yaml` 与 fair 系列 configs
  （`fair_causal_target10`/`fair_endpoint10`/`fair_dense10` 接近论文十帧协议）。
- 评估与诊断脚手架：`evaluation/`、按 `label_valid` 监督帧打分、前背景 Dice 口径、
  `reports.compare_runs` 跨 run 比对、anchor/prompt/flow deltas 诊断思想（可迁移为
  γ/CV/复合误差诊断的工程模板）。
- UNeXt backbone 代码可作 **随机初始化对照** 或 anchor 编码器备选，但不是 DIAG 的
  预训练内容锚；`DPFRPromptModulator` 的门控初始化套路（gate_init=-2.0 近零启动）
  值得新 HDC 头沿用。

## 3. 递交给 imp 的核对清单（冲突处以论文为准并记偏差）

按 Anchoring → PCLF → HDC → 损失/decoder → 指标顺序：

- [ ] **Anchoring**：定预训练 segmenter 基座（SAM/MedSAM/SAM2 系列二选一，记录版本与权重来源）；
  实现 $F_{tf}$ / $F_{tc}=\mathrm{Down}(F_{tf})$ 双尺度输出（Eq.5）；
  细尺度走内容路径、双尺度供观测编码。偏差备选：暂用 UNeXt 随机初始化时必须记为偏差（影响 H1）。
- [ ] **PCLF**：新建隐状态模块，禁复用 `DPFRDualPromptEncoder`（全窗双向，与因果冲突）；
  实现 $O^q$ 观测编码、$f^q_\phi$ 卷积向量场、显式 Δt Euler 单步（Eq.7）、$S^q_1=O^q_1$、
  校正门 $K^q_t$（Eq.8–9）；数据侧提供真实帧间隔 Δt（缺失/抖动采样见 Fig.5 清单）。
- [ ] **HDC-A**：$\gamma_t=\sigma(\cdot)$ 恒正门控单独成头，禁复用 FiLM `(1+scale)`；
  保留 H4 诊断钩子（逐层 γ 存档，供 CV/复合误差计算）。
- [ ] **HDC-B/C**：dense prompt 头（输出叠加到 $P_0$）、region token 头（K query 注意力，
  $\alpha_{c,f}\in(0,1)$ 门控并入 $Q_0$）；DPFR 的 flow warp 头与 logits 融合头**删除**，
  不得并存（架构冲突）。
- [ ] **训练/推理**：关闭 `mask_prompt_train.use_gt`（去 GT 泄漏）；decoder 加 J 候选+质量选择；
  损失按 Eq.17 六项实现，λ 待 human 补补充材料，先占位可配；
  保留 stopped-grad 短程 recurrence 与 identity-init refinement（如 basemodel 支持）。
- [ ] **指标/诊断**：Dice + surface-based HD95（统一实现待定）+ centroid drift/area roughness/
  inter-frame Dice variation；诊断脚本：相位 R²、状态干预 drop、γ CV、有序/shuffle 复合、
  Fig.5 采样压力（5 帧/3 帧/70%缺失/40%抖动保留率）。
- [ ] 每步记录偏差：凡与本清单冲突而沿用 DPFR 旧件处，按"论文为准、代码记偏差"写入实验记录。

## 4. 数据集/采样/seed 现状备注
- 论文口径见 000 文档 §5（十帧默认、patient-wise、seed 未公开）。
- upstream 有 `configs/data/{echo,camus,cardiacuda_*,domain}.yaml` 与 fair 系列实验 config，
  imp 下一步需核对十帧索引规则与 seed 是否与论文一致，不一致则记偏差。
