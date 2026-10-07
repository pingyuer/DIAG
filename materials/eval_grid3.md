# 格 3 ds 真间隔验收（eval n-20261003-162530-dcd ← impl n-20261003-092752-cbc）

口径依据：006 §2格3（真 ds 双轨；gap≤0.3pt 或 stride 变化<5% → 时钟项证伪）+ P3。

## 1. 结论：证伪（P3 死；ds 为恒等摆设）

- mlflow `grid3-ds-real` FINISHED（cc8dad3a，ckpt=grid1_s0_best）：stride2 gap +0.0000，
  stride3 gap +0.0002，drop50 gap +0.0004——max_gap 0.0004 << 0.3pt 线。
- 本地 ds 探针（grid1_s0 ckpt，真权重）：正常 clip ds≈0.84–0.94，静止 clip ds≡0.9779 全等，
  floor 0.0——ds 对"有/无运动"**零区分**；token 能量 E|d|≈0.0035，w·E≈−0.10，被 bias 0.51 淹没，
  softplus 全程工作在线性区上沿的常数段。
- 机制：`tokens()` 先 `.abs().detach()`，再 `metric` 线性层权重均值 −0.038（std 0.012，
  相对初值 0 几乎未动）——ds 头训练 30ep 实质未学习，能量输入本身又被 backbone 归一化压到 1e-3 量级。
  ds≈σ⁺(bias)+噪声，PCLF 吃到的 dts 恒等。

## 2. 实测复核

- 双轨实现读验：railA 真 `ds_head(f_tf)` vs railB ones ✅；test50 全量 patient-avg ✅；
  mlflow 四场景×双轨 8 指标齐（full/stride2/stride3/drop50），static 只打印未记库（记一行）。
- railA_full 0.9123 vs 本体重跑 clean 0.9123 一致；ckpt=grid1_s0（val 0.9120 世代）✅。
- code 谱系：run tag d5053945 vs 本地 HEAD 已漂（后继提交）；脚本 commit 55e9362 冻结可追溯。

## 3. 与论文/H 结论的对应关系

- P3 证伪入诚实章节：Euler 显式 Δt 路径活（PCLF 验收既有）但 ds 头供给恒等——"连续动力学时钟"在我方实现中不存在。
- 后继方向（二选一）：① 删 ds 头回 ones（承认 Euler 退化为残差更新）；② 重设计能量信号
  （backbone 特征归一化后差分太小是根因，delta_scale=8 不够或位置放错）。
- 不产生 Table 对照数值（离线诊断）。
