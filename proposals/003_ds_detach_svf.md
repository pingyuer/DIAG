# 003 重构提议：ds 头 + 单侧 detach + SVF flag-off 分支

依据：`proposals/001_diag_innovations.md`（DPFR 差距）、`proposals/002_mlflow_windows.md`
（W-step/W-epoch/W-event + O1–O10）、attachment 讨论（ds 驱动 / UNeXt 降级 /
SVF 硬约束 / detach 纪律）。只整理、不实现。
幂等核对：`proposals/` 下 000/001/002 均存在，无 003，本提议不重复。

立场：本提议是**强力重构**，不与论文 Eq.5–17 强绑定。author 授权：可重构实现，
评估模块隔离增益后直接接进去，不做 stack-of-trick。基线：CAMUS 全链 test
Dice 0.9132 / HD95 9.79（随机初始化基座已知偏差）。

## 1. ds 头设计（先上，零改前向签名）

现状：`src/diag/pclf.py:ScaleFlow.forward(feats, dts)` 已接受 `dts: Tensor(T-1)`，
但 `DIAG-code/train_camus.py` 喂的是 `torch.ones(9)`（无真间隔偏差）。
`src/diag/losses.py` smooth 项有 `/dt.clamp(min=1e-6)`，ds→0 会炸，先设下界。

设计：
- 输入：backbone 全局 token 序列 `F:[B,T,D]`（pool 到 token 后再差分，不在全分辨率上做）。
- `delta_F = F[:,1:] - F[:,:-1]` → 单层 Linear(`metric_head`) → `softplus` → `ds:[B,T-1]`，
  严格非负。batch 内向量化，无 CPU `if-else`。
- `bias` 初始化偏到 `softplus⁻¹(1)≈0.54`：ds 初值≈1，保持与现状 ones 的恒等延续。
- `ds = ds.clamp(min=1e-3)` 再喂 Euler 与 smooth 分母（防爆）。
- ds 经 backbone 的梯度第一版 **`detach()` 断开**：只当步长乘子，不当监督信号，
  避免二阶耦合。
- 工程心法：探头停顿/暗区 → ΔF≈0 → ds→0 → Euler 步长归零，"时钟免疫"。

验证：002 W-step 加 `ds_mean`/`ds_min` 两列；单 clip 过拟合先看 ds 不塌零；
变间隔/缺失采样（Fig.5 清单）是 ds 的终验。

## 2. 单侧 detach（顺手做，不动结构）

现状：`src/diag/hdc.py:RegionTokenHead` 中 query 参数与 S 两边都通梯度，
下游 Loss 直灌 S 会把状态带偏（单帧噪声捷径）。

设计：
- S 进 attention 前**单侧 `detach()`**（先断 S→query 捷径），不动结构、不改签名。
- `nn.Parameter` 禁止与 EMA 同写一张量：独立原型池（16×256 槽位）若做，
  用 `register_buffer("proto_run")`，train 写 buffer、eval 只读，
  `get_canonical_memory()` 返回 buffer 本身（无梯度，无需再 detach）。
  **本次不做独立原型池，另起分支。**

## 3. SVF flag-off 分支（加法分支，γ 不动）

动机：DPFR 自由 flow（tanh 截断无约束 dense flow）在低纹理超声边缘跑飞；
DIAG γ 通道门控又太弱（只调通道，不动空间）。SVF 是中间道路：
真 diffeomorphism 空间 warp，但小、轻、可关。

设计：
- UNeXt **降级为头**：只出 2 通道平稳速度场 `v:[B,2,H,W]`（小方差 init，近零邻域），
  1/4 分辨率预测再上采样 phi，省显存。
- `DiffeomorphicTransform`（Scaling&Squaring，6 步）出 `phi`；逆变换送 `-v` 同函数，
  零求逆开销。`F.grid_sample` 底层 CUDA 算子。
- 三件套缺一不可：`max_disp` 截断 + smooth 正则 + 小方差 init；否则复现 DPFR 跑飞。
- `align_corners` 全链统一（DPFR 用 True，别混）；`identity_grid` 按 shape 缓存，
  不每 forward 现建。
- **γ 通道门控 + H4 诊断（γ CV~1e-6 / 有序-shuffle 2.13×）不动**；SVF 当第四分支，
  `flag` 默认关。转正标准：warp 分支 `flow_prompt_delta > 0`（沿用 DPFR deltas
  诊断思想），否则删除，不堆 trick。

## 4. 评估协议（防 trick 堆叠）

- 对照：frozen 预训练 backbone（DINO-Small/MobileViT，权重冻结）+ frame-only 基线，
  锁死 seed / split / compute budget。UNeXt 不再当全能主干。
- 加法顺序：ds → SVF → prototype buffer。每个只看**隔离增益** + 两两交互；
  无增益即删。
- HD95 先崩即停（Dice 会骗人，论文 §4 明示 overlap≠boundary）。
- 诊断三项不过不谈群：ds 不塌零 / 恒等 phi 下 Dice 不变 / `gamma_cv` 量级。
- 偏差逐条进 mlflow tag；associative scan 不做（T=10 时 Python for 开销忽略，
  全视频 T>100 再谈）。

## 5. 给 implementation 的落地清单

1. `src/diag/ds_head.py`（新）：全局 token 差分 + Linear + softplus，bias≈0.54，
   输出 detach + clamp 下界 1e-3；`train_camus.py` 把 ones 换成 ds 头输出。
2. `src/diag/hdc.py`：region 路 S 单侧 detach，一行改动。
3. `src/diag/svf.py`（新）+ `DiffeomorphicTransform`：flag 默认关，1/4 分辨率，
   三件套，转正标准写进 run tag。
4. 002 W-step 加 `ds_mean`/`ds_min`；阈值沿用 O1–O10，新增 ds 塌零告警。
5. 不做：独立原型池、associative scan、backbone 替换（另起提议）。
