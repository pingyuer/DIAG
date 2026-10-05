# 010：论文向对照（GDKVM 调研 + CAMUS 主对比）

依据：`proposals/000 §3`（Table 1 口径与数值）、`proposals/001 §2`
（GDKVM/DPFR 代码差距）、`proposals/009 §1/§3/§7`（baseline 冻结 + 结果表 +
对比方法两类）。只整理、不实现。
幂等核对：`proposals/` 下 000–009 存在，无 010，不重复。
工具链备注：announce [1]（2026-10-05）要求代码对照走 lsp + ast_grep；
本仓 LSP 无 server（`xd://lsp status`：No language servers configured），
本次用 `ast_grep`（`class $NAME` 全枚举 `upstream_BanditPM/model`）+
原文读码；待 LSP 配好后重走 definition 覆盖关键符号。

## 1. GDKVM 调研（`upstream_BanditPM/` 只读，不动代码）

### 1.1 架构（`model/gdkvm01.py:242 GDKVM` 实测）

- ImageEncoder：ResNet{50,18} 预训练（`pretrained=True` 默认），取
  layer1–3（f4/f8/f16）；MaskEncoder：image+mask 双通道 ResNet18 + fuser +
  sensory updater（`MaskEncoder.forward`，`deep_update` chunk 可选）。
- KV memory + gated delta rule（`modules/gdr_core.py:13 GDRCore`）：
  state `S∈[B,N,K,C]`（K=64 key 维，C=256 value 维，8 head）；
  更新 = delta-rule 擦除-写入 + `−exp(A_log)·softplus(a_proj+dt_bias)` 衰减
  （`gdr_core.py:86-100` 实测三行 einsum：eraser/new/alpha-old）。
- 读出链：key/proj（KPFF，`model/kpff.py:28`，JIT recurrent）→
  `pixel_fuser` → QueryTransformer（3 block × 8 head × 16 query，
  `transformer/object_transformer.py:74`）→ MaskDecoder（16→8→4 上采样 +
  sensory GRU 更新）→ `AuxComputer`（sensory/q 双 logits）。
- 轻量对照另有 `model/memory_readout.py:8 MaskAwareMemoryReadout`
 （UNeXt-DynaKey 用，4 slot EMA + 余弦读，非主链）。

### 1.2 与 DIAG I1 的对立点（逐条）

| # | DIAG（Eq.2–4，角色分离） | GDKVM（Eq.1 式共享状态） | 对论文叙事的意义 |
|---|---|---|---|
| O1 | $x_t$ 内容锚与 $z_t$ 动力学分离，$\hat Y=D(\rho(g)x)$ | 历史内容、当前外观、结构先验全进 `state_BNCC` 同一张量（`gdr_core.py:100 old+new`） | 立论反面实例：Table 1 GDKVM 行即"共享表征能到多少" |
| O2 | 变换只调通道/ prompt/query（无 warp） | 记忆读出 + `pixel_fuser` 融合 + decoder 上采样，跨帧信息走 value 混合（einsum 加权） | 机制对照：상태混合 vs 增益调制 |
| O3 | 在线因果 PCLF（显式 Δt，$S^q_1=O^q_1$） | `temporal_access: recurrent`（fair config），state 跨帧常驻 `state_BNCC`（`.clone()` 递推）；另有 `use_first_frame_gt_init=True`（首帧 GT 初始化——DIAG prompt-free 侧无此物） | 首帧 GT 是最大协议差：对照跑必须关（`use_first_frame_gt_init: false`）或单列 prompted 档 |
| O4 | γ 恒正对角群 + 三项诊断 | 无群约束；`beta/sigmoid` 门 + `alpha` 衰减全可负/可变（`b_proj/a_proj` Linear 无约束） | H4 在 GDKVM 上无对应物——只比 Dice/HD95，不比诊断 |

### 1.3 KPFF/GDR 私有结构（imp 对照跑黑盒即可，不拆）

KPFF（JIT recurrent 融合）、`FeatureFuser`/`PixelFuser`、`MultiscaleSensoryUpdater`
（decoder 内 GRU）、`ObjectSummarizer`（16 summary + area 归一）——均为 GDKVM
私有件，DIAG 主干永不复用；对照跑只调 config，不读源码。

## 2. 主对比（CAMUS 先行，论文向）

### 2.1 Table 1 自动块本地可跑性

| 方法 | 本地可跑 | 判定 |
|---|---|---|
| 3D U-Net / ConvLSTM | 否（无代码） | 引论文数 |
| GDKVM | **是**（舱内 `upstream_BanditPM` + `gdkvm_*fair*.yaml`） | 同协议跑 |
| MemSAM | 否（链路未确认） | 引论文数（CVPR 2024） |
| OSA | 否（arXiv:2603.26188，无代码） | 引论文数；"弱群 vs 强流形"叙事对照 |
| SAM2-memory | 否 | 引论文数 |
| DIAG（我方） | 是（R0-6 a1 待跑） | baseline_v10 + R0-6 |

### 2.2 GDKVM 同协议跑法（imp 入口）

- 配置：`configs/gdkvm_camus_fair_dense10.yaml`
  （`prediction_mode: dense10` + `temporal_access: recurrent` +
  `protocol_version: v3_canonical_no_leak`）为起点；注意其
  `backbone_pretrained: false`（fair 限定）vs 主 config `gdkvm_camus.yaml`
  另有 eval 档（阈值 0.30–0.75 step 0.01 + hflip-TTA + 后处理 largest/fill/
  remove-small/min_size 8/closing 关——与我方 006 后处理口径天然对齐）。
- 必须改：`use_first_frame_gt_init: false`（去首帧 GT，与 DIAG prompt-free 对齐；
  开/关各跑一行则多一个 prompted 参照档，可选）。
- 容器内单独 clone 跑（GDKVM 不进 `src/diag/`、不进 DIAG 仓，spec §3 约束）；
  同 split（md5 `682f3d89`）+ 十帧 + patient-wise；产物只回传 metrics 行
  （Dice/HD95 双口径 + 切片三件），ckpt 不回传。
- experiment 名独立（如 `gdkvm-compare`），与 `diag-camus-train` 命名空间隔离；
  evaluator 只收数字进表。

### 2.3 其余方法引用格式（method 章用）

一律标"论文引用数（协议未对齐）"：方法名 + 出处 + CAMUS Dice/HD95 原数
（000 §3 表）+ 一句话定位（记忆派 / FiLM 派 / 配准派 / 解耦派，001 §2 叙事）。
禁与舱内裸口径并表；并表只允许 C1–C4 + R0-6 同协议行（009 §7 规则延续）。

## 3. 代码归属（spec §3 落地）

- GDKVM 对照代码永不进 `src/diag/`（DIAG 主干）与 DIAG 仓；跑法 =
  远端容器内单独 clone（`wangrui2025/GDKVM` 或舱内 upstream 二选一，
  优先舱内——split/config 现成）+ task 队列制（hostname-pin，同 009 §6）。
- 对照 run 记独立 experiment（`gdkvm-compare`）；mlflow tags 加
  `protocol: fair_dense10` + `first_frame_gt: false` + split md5。
- 跨域（Adult/Pediatric/CardiacUDA）本次不做（G2 gate 另起节点，004 §4 延续）。

## 4. 刷点线（CAMUS 内，009 §5 R0 后）

R0-6 s0 FLAKY 追查 → ce×rec 细化（s2 胜者邻域小网格）→ MedSAM 真 P0/Q0
（a2 阻塞中：权重指针未下载 + 1024×4 OOM）→ 高分辨 decoder 128（a3 未立项）。
网格 + 双 seed（009 §4 纪律延续）。本提议不展开刷点设计，只定顺序；
R0-6 先落（队列队首，009 §6 已排），刷点另起 proposal。

## 5. 给 implementation 的落地清单（按序）

1. GDKVM fair 同协议跑：`gdkvm_camus_fair_dense10` + `first_frame_gt=false`，
   容器内单独 clone，队列 task 2 个（s0/s1），产物只回传 metrics 行。
2. R0-6（a1）DINO×ce0.05/rec1.0 双 seed（009 队首延续，本项并行不抢卡：
   不同容器各一）。
3. 引用数表：其余五方法按 §2.3 格式整理进 `materials/compare_table.md`
   （纯文档，evaluator 收数）。
4. 不做：跨域、MedSAM 权重下载（human）、高分辨 decoder、SVF 重启。
