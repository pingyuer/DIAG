# 005 提议：优化修复先行 + 基座替换两轨设计

依据：`proposals/001`（DPFR 差距）、`003`（重构）、`004`（压力/SVF/聚合/跨域/搜索）+
验收沉淀：`materials/dualrun_h1.md`（H1 缺口 11.4pt val / ds-full +0.8pt 非隔离 /
HD95 test 10.64 vs 40.30 / ds 恒 1.0 / K→0.886 / best 分化正常）、
`materials/svf_grid.md`（18 点程序性删除、disp 倒 U 非噪声、欠训区 6ep/60 未排除）、
`materials/eval_004.md`（P1–P4 保留率方向一致 −1.3~−2.8pt / P4 绕过 ds 头自限 /
G2 零样本 0.0394 gate-stop）、λ 网格 impl 结论（14 点 0.53–0.72 vs 基线 0.90 全塌；
排序 ce-0.01 0.724 > rec-1.0 0.708 > iou 0.70 > flow 0.64–0.66 > smooth 0.62 毒药 >
dice 降权死 0.03；seed7 复跑 −8.6pt 方差；HD95 全崩作废；`loss/*_ep` live 可比）。
只整理、不实现。幂等核对：`proposals/` 下 000–004 存在，无 005，不重复。

立场延续 003/004：强力重构授权有效；一次只加一个可关分支；无隔离增益即删；
HD95 先崩即停；禁手调单点报喜。

## 1. 优化修复先行（便宜，先做；λ 网格塌是第一现场）

λ 网格 14 点最佳 0.724，距 30ep 基线 0.90 差 ~18pt——但网格是 60 子集/6ep
欠训区短预算（seed 方差 −8.6pt 同量级），"全塌"判据不足，只能定性读排序：
ce 降权有效、rec 升权有效、smooth 有毒、dice 降权即死。这说明**量级失衡**，
不是"哪个 loss 没用"。

### 1.1 loss 尺度归一（先做，不跑网格也先做）
- 现状：六项裸值相加（ce~0.18/dice~0.60/rec~0.09/smooth~0.006/flow~0.09/iou~0.69，
  windows_002 快照），smooth 天然小 2 个量级——λ=0.01 的 smooth 实际贡献
  ~6e-5，等于没有；而 dice~0.6 主导，降权即死符合预期。
- 设计：每项除以其 **running 均值**（EMA，`detach`，首 epoch 用首 batch 快照），
  再乘 λ。λ 语义从"裸值权重"变为"相对重要性"，网格从量级搜索降为重要性搜索。
- 落地：`DiagLoss.forward` 返回值不变，加 `loss_normed` 字典 + W-step 记录；
  归一开关 flag 默认关，先做单 run 对比（归一 on/off 同 seed），隔离增益为正
  才进网格。

### 1.2 动力学梯度饥饿（ds 恒 1.0 + K→0.886 的联合诊断）
- 现象：ds 全程恒 1.0（240/240 点，什么也没学）；K 0.515→0.886 向观测侧漂移
  （O3 观察项，未触发 0.95 动作线）。联合解读：Euler 步长恒等 + 门偏向观测 =
  PCLF 退化为"观测直通 + 小残差"，动力学分支拿不到有效梯度。
- 排查序（按便宜到贵）：
  1. `ds_head` 在 `torch.no_grad()` 下调用（train_camus 现状）→ ds 参数零更新，
     恒 1.0 是**结构必然**，不是"学不动"。先放开 ds_head 进优化器（独立 lr，
     建议 1e-4），单 run 看 ds 是否动；动了再谈时钟语义。
  2. smooth 压制：smooth/Δt 惩罚相邻状态差，K→1 时状态≈观测、相邻差大，
     smooth 反压 VectorField——smooth 有毒（0.62）与 K 漂移可能是同一机制两面。
     做 smooth {0, 1e-4, 1e-3} 三点消融（归一后），看 K 回落与否。
  3. detach 纪律复查：ds tokens 已 detach（003 正确）；region S 单侧 detach
     只断 query 捷径，gate 路仍通 S（003 验收确认）——若 K 继续漂移，
     查 `ObsEncoder` vs `VectorField` 输出范数比（观测强 10× 则门自然偏观测，
     需能量平衡而非调 λ）。
- 禁止动作：在 ds 放进优化器之前，任何"P4 抖动保留率 1.0 证明时钟免疫"的
  表述继续禁用（eval_004 记录 P4 已自限：绕过 ds 头 + ds 恒 1.0）。

### 1.3 单 loss 消融顺序（归一落地后，短预算）
- 顺序：smooth-off → flow-off → rec-off → iou-off → ce/dice 二选一降权，
  每次只关一项，同 seed/split/budget，60 子集/6ep（与 λ 网格同预算可比）。
- 停止规则：关某项后 val 上升 → 该项有害，提议删除（走 003"无增益即删"）；
  下降 <2pt → 冗余，记观察；下降 >5pt → 核心，λ 网格重点扫它。
- smooth 优先（毒药嫌疑最大），rec 其次（升权有效，需确认是"有用"还是
  "代偿 dice"）。

### 1.4 HD95 远差（test 10.64 vs 论文 5.43，约 2×）
- 现状：decoder 输出 64×64（256 输入的 H/4），`eval` 上采样计分；mask 经
  双线性 warp/上采样糊边界是已知机制（003 讨论）。
- 二选一（只做一个，先便宜的）：
  (a) boundary loss：加 Dice 的边界变体（surface 距离加权 BCE，权重进 λ 网格
  第七维，初值小 0.01 起）；(b) 高分辨 decoder：decoder 输出提至 128×128
  （多一级上采样 + refine，参数小），GT 对齐仍 nearest。
- 验收：只看 HD95 降 + Dice 不掉（掉 >1pt 即停，002 O7）；HD95 口径先统一
  自研实现与论文 surface-based 的差异声明（全链验收偏差 ⑤延续）。

## 2. 基座替换（大开大合；优化修复无结论前不动手）

门槛：§1.1–1.3 任一产出"归一后基线回 0.85+"或"某 loss 确诊有害"后，才启动
基座替换。随机初始化 UNeXt 下讨论基座是空对空。

### 2.1 (a) MedSAM 风：预训练 encoder + 真 P0/Q0
- 得：HDC-B（dense prompt 叠加 $P_0$，Eq.13）、HDC-C（region token 并入 $Q_0$，
  Eq.14–15）第一次名副其实；内容锚表达力补齐（H1 缺口解释力）；论文 Eq.5–17
  口径对齐度最高。
- 代价： MedSAM/SAM 权重下载链路（远端无外网？先查 `~/.cache/huggingface/hub`
  为空——权重怎么进容器是第一任务）；encoder 重（A30 24G 够，但 batch 可能
  从 2 降）；接口改造（$P_0/Q_0$ 抽取点需读权重代码）。
- 接口冻结：`(F_tf, F_tc)` 形状语义不变，`ContentAnchor` 只换 backbone 实现，
  PCLF/HDC 不动（003 §4 约束延续）。

### 2.2 (b) DINOv2 风：自监督 dense 特征做锚 + 观测
- 得：DINO patch token 天然 dense 对应 + 语义平滑（003 附件论证），超声
  speckle 下比随机 UNeXt 强一个量级；轻（DINO-Small frozen 只前向）；
  ds 头的 ΔF 噪声 floor 降低（当前 pooled 能量 O(0.05) 需 delta_scale=8 放大，
  好特征下该 hack 可删）。
- 代价：无 $P_0/Q_0$（HDC-B/C 需重推导：dense prompt 改为"加到 F 上"而非
  "叠到 $P_0$"，region token 改为"拼到 Q 前"而非"并入 $Q_0$"——这是论文
  口径 deviation，需记）。
- SAM2 约束：只取 image encoder，不取 video memory（memory 与 I1"反对共享
  时空状态"叙事冲突，001 §2.1 已定 GDKVM 只当对照基线）。

### 2.3 两轨决策
- 默认单轨 (b) 先行（便宜、链路短）；(a) 当且仅当 human 确认权重可进容器
  才并行。frozen backbone + frame-only 基线锁 seed/split/budget，加法验收
  （003 §4 原样）。backbone 本身不进 λ 网格，单独节点。

## 3. 搜索纪律（沿用 004 §6，加两条补丁）

- 网格全记（params + code_sha + split md5 + seed）+ 最佳点独立 seed 复跑，
  禁手调单点引用——λ 网格 seed7 −8.6pt 已证明短预算方差吃掉一切小增益。
- 补丁 1：短预算网格（60/6ep）结论一律加"欠训区"限定词；转正/删除 verdict
  只认 30ep 全预算（SVF 删除是程序性 verdict，svf_grid §1 已自限——同例）。
- 补丁 2：`loss/*_ep` 六项 epoch 均值已 live（λ 网格补上），后继网格间可比；
  搜索预算写死 max runs，超停。

## 4. 给 implementation 的落地清单（按序）

1. §1.1 归一开关 + 单 run on/off 对比（不动网格）。
2. §1.2-1 ds_head 进优化器（独立 lr 1e-4）+ ds 是否动的单 run 判定。
3. §1.2-2 smooth 三点消融 + K 回落检查。
4. §1.3 单 loss 消融序列（smooth → flow → rec → iou → ce/dice）。
5. §1.4 boundary loss (a) 先行；掉 Dice >1pt 即停转 (b)。
6. §2 门槛判定后才开基座轨；(b) 默认，(a) 等权重链路。
7. 不做：SVF 重启（已删 verdict，除非 §1 结论翻转）、原型池、scan。
