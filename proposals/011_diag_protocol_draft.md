# 011：DPFR 起点协议草案 + CAMUS 闭环 + 第一组对照

依据：`docs/project/diag_project_alignment.md`（2026-10-06 方向锁定）+
`docs/project/dpfr_diag_protocol_manual.md`（DPFR 双 seed 实际执行核对）+
用户 2026-10-06 六节指示 + 011 spec（DPFR 实际数据流为起点）。
只设计、不实现。幂等核对：`proposals/` 下 000–010 存在，无 011，不重复。
标记约定：**[定]** = alignment/手册已确定；**[建议]** = proposer 推荐值，
需证据或用户裁定后转正；**[问]** = 需用户裁定的实质问题，集中 §4。

## 1. CAMUS 最小闭环（先行）

### 1.1 数据版本 / 切分 / 标签源帧映射

- **[定]** 沿用用户已冻结切分，不重新划分（alignment §任务数据流；手册 §3）。
- **[定]** 当前已知切分：`camus_public_datasplit_20250706.json`，
  train/val/test = 400/50/50，切分 MD5 `682f3d8978596f040e8d6cb1ef46cbc7`
  （手册 §2.1）。文件名日期不证明即用户新切分，以内容和指纹对用户切分为准。
- **[定]** 数据目录（手册 §2.1）：`/input0/processed/camus_png256_10f`，
  `img/<patient>/*.png` + 同名 `gt_lv/<patient>/*.png`，每病例十帧配对无缺口
  （两端容器已检查）。**[问]** 用户已重新制作切分（082731-0d2 spec）——
  若与上述文件不同，只需给 manifest 路径（§4 Q1），不重跑历史核对。
- **[定]** 标签源帧映射缺失：挂载目录无 `metadata/`，ED/ES 身份、原始帧编号、
  时间间隔、spacing 均不可恢复，加载器 `frame_indices` 回退 `0…9`
  （手册 §2.2）。重跑前必须恢复每病例源帧映射 + 每帧 `label_source`
  （人工/传播/插值/缺失），十帧有掩码文件 ≠ 十帧同种人工监督。
- **[定]** 预处理契约（手册 §4.1，每病例片段冻结字段）：
  dataset/patient_id/video_id/view、split+指纹、source_video/source_frame_ids、
  timestamps-or-fps、ED/ES（原始帧身份+局部位置，禁默认首帧=ED）、
  label_source、original/resized/spacing、repeated_source_frame、
  preprocessing_version（含归一化+插值规则）。
- **[建议]** 初始主实验固定 ED–ES 端点间十帧片段，按原始时间排序
  （手册 §4.1）；映射未恢复前不重新生成切分，不宣称复现采样过程。

### 1.2 采样窗口 / 输入 / 增强

- **[定]** T=10 固定同一组十帧，源帧可追溯；输入 256×256（当前 DPFR 已完成
  运行配置，不自动扩展为全域统一要求，alignment §实验组织）。
- **[定]** 图像灰度 uint8→float/255，模型内 `(I−0.5)/0.5`；掩码 `mask>0` 二值 LV
  （要求输入已是 LV 二值掩码，手册 §2.3）。
- **[定]** 训练增强（手册 §2.3，当前展开配置）：整段共享几何参数，
  hflip p=0.5 + 旋转 ±10°（图像 bilinear / 标签 nearest）；无额外强度增强。
  验证/测试固定处理，无随机窗口/增强。
- **[定]** batch 输入契约（手册 §4.2）：images [B,T,C,H,W] + source_frame_ids
  [B,T] + dt [B,T−1] + frame_valid/pair_valid；supervision masks +
  annotation_valid/loss_valid/eval_valid + label_source；metadata
  （patient/video/view/ED/ES/空间+版本）。三有效集合可不同；
  无标注 masks 零占位但禁监督；`num_objects=1` 固定声明（禁由 GT 前景定类别数）。
- **[定]** 重复/填充规则训练前冻结：恢复映射后去重 + 显式填充；
  禁把零 dt clamp 成小正数隐式处理（手册 §4.2）。`pair_valid` 排除跨界/
  填充/重复源帧时间对。

### 1.3 前向可见性 / Ω 监督 / loss

- **[定]** 自动离线视频分割：$I_{1:T} \to \hat Y_{1:T}$，$\mathcal L$ 只在标注帧
  Ω 上算；全窗可见（含对某输出帧的后续帧），预测覆盖全部输入帧
  （alignment §任务数据流）。
- **[定]** 前向禁 GT 点/框/掩码提示与首帧 GT 初始化；训练 GT 只进监督；
  自预测 mask prompt 属内部表示可用（alignment §任务数据流）。
  当前 DPFR：`use_gt=false` + `detach_mask_prompt=true`，GT 替换分支全关，
  验证/测试 GT prompt 比 0 / anchor prompt 比 1（手册 §2.4）。
- **[定]** 当前 DPFR 数据流六步（alignment §实现锚点，`dpfr/model.py`）：
  逐帧 UNeXt anchor → mask prompt（detach）→ 全窗 Transformer（无因果 mask，
  不消费 dt）→ 多尺度调制 + 共享 decoder（gated_add）→ 2D 位移 warp
  （grid_sample，**空间网格位移 ≠ DIAG latent flow**，手册 §2.5）→
  门控融合 anchor/prompt/flow。
- **[定]** 当前 DPFR 损失（手册 §2.6，有效标注帧上，四路各病例先平均再 batch
  平均）：final 1.0 / anchor 0.3 / prompt 0.3 / flow 0.2（CE+Dice 各路）+
  位移幅度 0.005（全帧）/ 空间平滑 0.01 / 相邻位移差 0.01；
  点监督 12544 点/oversample 3/重要性 0.75（损失位置采样 ≠ 人工点击提示）。
  `total_loss` 只累加 `dpfr_` 项（通用 loss_ce/dice 另算，禁重复解释）。
- **[建议]** DIAG 重跑 loss（手册 §6）：Ω 病例归约
  $L_{seg} = \mathrm{mean}_b\,\mathrm{mean}_{t \in \Omega_b}$；
  图像辅助项有效集合：重建→全 frame_valid（源码现只用标注帧，改即方法调整，
  须与公式/消融/版本同步），平滑/下帧预测→pair_valid；下帧目标 stop-grad/
  冻结/EMA 显式声明（现无 detach，切换即不同训练定义）。

### 1.4 训练预算 / 优化器（推荐值，标[建议]）

- **[定]** DPFR 已验证起点（手册 §2.7，不自动成全底座最优）：batch 2 视频
  [2,10,1,256,256]，4000 iters（≈8000 片段呈现 ≈20 epoch），AdamW lr 1e-4
  （UNeXt 组 ×0.3 → 3e-5），warmup 300 + cosine（min 0.05），wd 0.001
 （bias/norm 免衰），grad clip 3，AMP 开，EMA 关，底座无预训练；
  seed {0,1} 但 `deterministic=false` + cudnn_benchmark（禁承诺逐位复现）。
- **[建议]** DIAG 主实验：30ep×400×batch2×lr3e-4（009 baseline 冻结延续，
  grid1 方差 0.2pt 已验证稳定）；seed {0,1} 双跑；外部 baseline 允许各自
  验证集调参（记搜索空间+预算，禁统一迭代数掩盖收敛差，手册 §7）。

### 1.5 val 选 ckpt / test / 阈值 / 后处理

- **[定]** 固定验证片段+集合，每验证点病例平均 Dice（先病例内平均有效帧，
  再对病例平均）；最高者当选，平局取早；训练结束显式重载再测；
  测试标签不参与选 ckpt/阈值/后处理/超参（手册 §8.1）。
  当前 DPFR 通用测试入口不自动重载最优（本次最优点恰为末步 iter4000，
  其他运行须显式加载，手册 §2.8）。
- **[定]** 主结果 raw 输出 + 冻结阈值 0.5（单 logit sigmoid / 双 logit 前景
  softmax；`>` vs `≥` 统一记录）；DIAG 多候选先用预测质量选（手册 §8.2）。
  TTA/连通域后处理只作附加结果，raw/TTA/post 三档独立记录，禁跨档比模型。
- **[定]** 阈值搜索只用验证集，同规则冻结后另建协议；当前 DPFR raw 0.5 为主锚点。

### 1.6 指标：Dice/IoU/HD95 单位 + 聚合 + 空 mask

- **[定]** 主统计病例平均（先病例内有效帧平均，再病例平均；多视图聚合固定）；
  每帧/片段/病例结果保留；ED/ES 子项按原始身份匹配，无身份只报 first/last，
  禁重命名（手册 §8.3）。
- **[定]** HD95 固定一种主实现 + 空间定义，另一实现留对齐记录；
  256 网格无 spacing 时单位 = 网格像素，禁标 mm / 禁对比论文 mm 数。
  GT/预测双空 Dice=1、单空 Dice=0；HD95 单空报告失败数 + 预定义聚合
  （如网格对角线罚值），禁静默删失败帧（手册 §8.3）。
- **[定]** 多 seed 保留原值+均值+样本标准差+运行身份；禁只选最好 seed；
  置信分析以患者为重采样单位（禁十帧当十独立样本）。
- **[建议]** 主实现选 MONAI mirror（hd95_align 已对齐上游；cdist 留交叉记录）；
  当前 DPFR HD95 缺失（禁从 Dice/IoU 推算），补齐走已批节点，不在本草案内。

## 2. 第一组对照（同底座同数据流；解耦检验）

约束（spec + alignment + 手册 §1/§10）：联合训练模型内部 anchor
（0.9308/0.9323，final−anchor 仅 +0.09pt）**禁作独立 baseline**
（它收全路梯度）；full window 与 causal 可同离线主表比较，但禁解释为
同在线信息下机制比较；在线主张另立因果实验。

| 组 | 配置 | 与完整 DIAG 的唯一差 | 检验的问题 |
|---|---|---|---|
| A | 独立训练单帧 UNeXt（逐帧编解码，无时序件） | 无时间访问 | 内容锚上限（H1 分母） |
| B | 同底座 + 素时序融合：逐帧特征 → 时间维均值池化 → 同一共享 decoder（无 mask prompt、无 Transformer、无 warp、无门控） | 时序混合无结构（均值是无参数融合） | 若 C≈B≫A：增益来自"看见多帧"本身；若 C≫B≈A：增益来自结构化条件（调制/warp/门控）——这才是解耦检验 |
| C | 当前完整 DPFR（anchor→mask prompt→全窗Transformer→gated_add→warp→门控融合） | —（起点本身） | 主干参照 |

- B 的设计理由：均值池化是"最弱时序利用"，参数量≈A，训练预算/增强/监督
  与 A/C 一致；B 与 C 差 = 全部结构化件（prompt+Transformer+warp+门控）的
  联合效应，C 与 B 差若显著再逐件拆（prompt-only / warp-off / fusion-off，
  另起消融，不在本组）。
- 预算：三组同 split/十帧/batch/4000iters/seed{0,1}；val 选 ckpt + test
  raw 0.5 + 病例平均；B/C 另报 anchor 分支诊断差（只诊断，不转正）。
- 口径：prompt-free 全组；Ω 监督一致；HD95 主实现同 §1.6；TTA/后处理三档独立。

## 3. 扩展规划（Adult/Pediatric/CardiacUDA，逐域六项，只规划）

每域落地前按 §1 六项逐项冻结（不展开，只列清单+已知）：

1. 切分/manifest + 指纹（用户冻结切分延续；当前仅 CAMUS md5 已知）。
2. 标签源帧映射 + `label_source`（EchoNet 系稀疏监督：手册 §4.4 例子——
   十帧仅局部 0/9 有人工标签时，三有效集合均为 [1,0…0,1]，中间零 mask 禁监督；
   验证只在两帧算真实 Dice，先两帧平均再病例平均）。
3. 采样窗口/T（CAMUS 十帧不自动扩展；各域固定分辨率可不同，但域内比较对齐）。
4. 输入尺寸 + adapter（底座通道适配/内部 resize/预训练归一化属 adapter，
   输出统一到声明评价空间；禁 adapter 内换帧/加注/改评价位，手册 §4.3）。
5. 监督 Ω + 损失归约（病例归约 §1.3；稀疏域十帧真实均值禁报无标签帧 Dice）。
6. 指标聚合（ED/ES 身份匹配；full cycle 另建片段清单，禁由端点距离猜周期，
   手册 §4.4）。

顺序建议：CardiacUDA 先试点（论文区分度最大，004 G2延续）→Adult→Pediatric；
mDice 只报全覆盖方法（000 §3）。

## 4. 未定参数（[建议]+理由）与待裁定问题（[问]集中）

[建议]清单（理由见括号；转正需证据或用户拍板）：
- s1：主实验 T=10 固定片段 + 256 输入（DPFR 已验证起点；跨域可异，域内齐）。
- s2：DIAG 主预算 30ep×400×batch2×lr3e-4 + seed{0,1}（009 grid1 稳定证据）。
- s3：HD95 主实现 MONAI mirror（上游对齐；cdist 留记录）。
- s4：主表 raw 0.5 + 病例平均（DPFR 主锚点延续；TTA/后处理附表）。
- s5：B 组用时间维均值池化（最弱时序利用，参数≈A，解释力最干净）。
- s6：J 候选/IoU 命名：`losses.py` 的 `inter/(prob+gt).sum()` 缺减交集
  （手册 §5.3）——重跑版修正分母或改名，二选一在本草案定（建议修正分母，
  与论文 IoU 定义对齐）。

[问]用户裁定（影响有效性、无证据，需拍板）：
- Q1：新切分 manifest 路径（082731 spec 称已重做；手册只见旧 md5）。
  无路径则 §1 切分项冻结在旧 md5+待核对状态。
- Q2：元数据链路（ED/ES、timestamps/spacing、label_source）是否有源可恢复？
  无则 §1.1 映射项永久记缺，论文相关主张降级。
- Q3：外部方法预算上限（各自调参与统一迭代数的取舍，手册 §7 未定）。
- Q4：训练 teacher forcing 变体是否保留（手册 §6.3 允许单列；默认主实验关闭）。
- Q5：验证集阈值搜索开不开（手册 §8.2 另建协议；默认不开，raw 0.5）。
- Q6：backbone/尺寸/采样是否在本轮冻结（announce #4 明确未冻结；本草案
  默认 CAMUS 先行，他域另议）。
