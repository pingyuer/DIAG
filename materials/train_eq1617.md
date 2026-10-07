# 训练/推理 Eq.16-17 验收（eval n-20260930-143403-a3a ← impl n-20260930-141014-389, commit e33213f）

口径依据：`proposals/000_diag_paper_parse.md` §1.4（Eq.16 J 候选+质量选择、Eq.17 六项损失、在线因果推理、下一帧仅训练监督）+ `proposals/001_diag_innovations.md` §3 训练/推理清单项（关 `use_gt`、decoder 加候选选择、λ 占位可配）。

## 1. 结论：通过（带三条如实记录，不打回）

- J 候选选择、六项损失可反传、全参数有梯度、无 GT 泄漏、无 flow warp/logits 融合件、远端 CUDA——父节点断言全部属实，本地独立重跑逐项复现。
- 记录 A（λ 占位）：`DiagLossWeights` 为 ce=1.0/dice=1.0/rec=0.1/smooth=0.01/flow=0.1/iou=0.5，`note` 字段明示 PLACEHOLDER 待补充材料（human）。当前 total=1.8946 只是占位权重下的 smoke 数值，**禁止**引用为论文对照值；补充材料到达后权重变更须重跑。
- 记录 B（重复 run）：exp96 内 `smoke-train-eq1617` 有两个同 sha run（7ebf5154 / 01b384dc，同 code_sha=ca71e401…，同 loss_total=1.894639）。与 PCLF/HDC 节点同类情况一致，系 smoke 脚本重跑；验收以后者（01b384dc）为准。
- 记录 C（token 扫描说明）：forward 四模块源码扫描中 `logits` 命中 5 处，经逐条定位均为"候选 logits/quality logits/BCE-with-logits"的中性术语与 hdc 注释中的架构要求声明，无 warp/grid_sample/use_gt/gt_prob/teacher 残留；四模块 forward 签名均无 gt 参数。判定无 GT 泄漏成立。

## 2. 实测复核（本地 .venv torch 2.14 cpu，真 clip 10 帧，seed 0，阈值伪 GT 仅作 loss target）

| 断言 | 实测 |
|---|---|
| masks=(10,1,3,256,256), quality=(10,1,3) | ✅ J=3 |
| quality 初值 proba≈0.5（最大不确定启动） | ✅ 0.5 |
| 六项损失全>0可反传：total=1.8946（ce=0.8739/dice=0.6278/rec=0.2458/smooth=0.0024/flow=0.2173/iou=0.6931） | ✅ 与 run 指标 1.894639 一致 |
| WTA best 前 5 全为候选 2（未训练快照，合理） | ✅ [2,2,2,2,2] |
| 推理 argmax quality 可选（(T,B) 索引） | ✅ 前 5 全 0（未训练，合理） |
| Omega 掩码生效：去掉 1 帧 total 变化；全空 Omega 断言 | ✅ True / `AssertionError: Omega empty` |
| 全链路 backward：pclf+hdc+dec+loss 无 grad=None 参数 | ✅ 0 |
| smooth 因果（仅 (i,i-1) 过去对）/ flow 用 S_t→F_{t+1} 且仅 loss 内 | ✅ 代码读验 + probe 形状 (96,96,1,1)×2 |
| recurrence g 零初值因果稳定（detach 过去） | ✅ 代码读验 |
| `git diff e33213f HEAD -- decoder/losses/smoke_train` | ✅ 空，无漂移 |
| 远端 :32237 CUDA decoder | ✅ masks (2,1,3,256,256) |

## 3. 设计核验（相对 001 §2.5 的冲突清单）

- 无 `mask_prompt_train.use_gt` 路径：新四模块压根没写 GT 输入分支，不存在"关闭"而是"从未引入"；spec 要求满足。
- 无 flow warp 头（tanh 位移场+grid_sample）与 logits 级残差融合头：grep 确认无 `grid_sample/warp` 实现。
- 调制点全在 decoder 之前；decoder 为共享 trunk 动态读出（非 J 私有 conv 栈），WTA 下 loser 经 quality 路径仍有梯度。
- 迟上采样 H/2→2x bilinear + 零初值 refine（step-0=bilinear）；rec 探头与 flow 探头内置于 loss 内，forward 保持 GT/next-frame free。

## 4. mlflow run（exp96）

- run `smoke-train-eq1617` ×2（见记录 B），`tags.node=train-eq1617`，`source.name=DIAG-code/smoke_train.py`，`metrics.loss_total=1.894639, n_params=648614.0`——与本地重跑一致。
- code_sha=ca71e401… 即当前 HEAD 链（e33213f 后仅 evaluation 提交，无代码漂移）。

## 5. 与论文结论的对应关系

- 本节点是 Eq.16–17 的结构/训练存在性验证，不产生 Table 1–2 对照数值。
- H1–H4 验收均需 CAMUS 数据管线 + 指标脚本（`diag_metrics.py` 节点）+ 真实训练后方可用；当前 total/WTA-best/argmax 均为随机初值快照，**禁止**引用。
- 缺失项延续：λ 真值（human 补补充材料）、预训练基座（影响 H1）、真实 dt 数据源、HD95 统一实现——四项均待后继节点，记于此。
