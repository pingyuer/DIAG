# 009：DINO-small baseline 锁定 + 最深 idea 三线任务树

依据：eval 全线结论（`materials/` 30+ 份）+ `proposals/000`–`005`、`006_decoupling_matrix`、
`006_post_eval`、`007_pipeline_cleanup`。只整理、不实现。
幂等核对：`proposals/` 下 000–008（含 006 双份、007）存在，无 009，不重复。
立场：本提议是**重立**——DIAG 是 author 在写论文，000–008 是第一轮试错；
H4（γ 群）、ds 时钟等假设弱是因为实验没跑够（格 4 三线阴性、格 3 ds 证伪均是
短预算/单配置阶段读数），不是结论已死。009 把散点收成"锁定 baseline + 三线
任务树"，后继按树跑搜索/复现/改进，把弱假设逐个做实或证伪，回流正文。

## 0. 最深 idea（全文统领，后续一切任务的归属判据）

单帧分割是基础锚 $x_t$；历史信息是独立建模对象——变换生成状态 $z_t$；
时序上下文只参数化内容上的可组合群变换 $g_t \in G$（Eq.2–4），群作用
$\rho(g_t)x_t$ 给出时序条件化表示。

架构映射（三线归属一切任务）：

- **估计器**（变换做准）：PCLF（ds-head 显式 $\Delta t$ + VF-deep 向量场 +
  K 门预测校正）+ HDC-A 恒正 $\gamma$。回答"$g_t$ 估得准不准"。
- **读出**（读出做全）：HDC-B dense prompt / HDC-C region token。回答
  "$z_t$ 读得全不全"。
- **锚**（锚做强）：ContentAnchor（DINO-small frozen + 适配）。回答
  "$x_t$ 强不强"。

归属规则：新任务先判属于哪线，再进该线队列；跨线任务拆成单线子任务
（单一变量铁律延续）。SVF（空间 warp）已程序性删除（svf_grid），
重启动需单独提案，不在本树内。原型池/scan 不在树内（003 已排除）。

## 1. Baseline 冻结（imp 起跑线，逐字锁死）

| 项 | 冻结值 | 来源 |
|---|---|---|
| anchor | DINO-small frozen + 1×1 适配 C=96（`--anchor dinov2-small`） | eval_008 P1 方向证实 |
| ds-head | 进优化器，独立 lr 1e-4（`--ds-lr 1e-4`） | 005 §1.2，grid1 ds 首次真动 |
| VectorField | VF-deep ON（`--vf-deep`） | eval_vf：10x 证伪、deep 有效 |
| RegionTokenHead | detach=ON（默认，不加 `--no-detach-region-s`） | 格 2：OFF Δ−0.2/+0.9pt 混杂，ON 为主干 |
| loss | ce×rec 联合 ce=0.05/rec=1.0，其余占位（dice=1/smooth=0.01/flow=0.1/iou=0.5/boundary=0） | stage-2 双 seed +0.5pt |
| SVF | 关（`svf_head=None`） | svf_grid 删除 verdict |
| 归一 | 关（不加 `--norm`） | opt_005：归一对比未转正 |
| seed/split/budget | seed {0,1} 双跑；split md5 `682f3d89`；30ep×400×batch2×lr3e-4 | grid1 方差终结 0.2pt |
| ckpt schema | 含 `ds_head` + `frame_only` 键 + md5 tag | 007 §2（dualrun 债务已关） |
| verdict 口径 | 主判据 mirror-HD95，Dice 参照；裸口径看模型、应用口径看表 | 008c 作废声明 + 006 判据 |

当前最优 ckpt：`outputs/s2full_s0/s1_best.pt`（UNeXt+联合权重，
val 均值 0.9169 / test 0.9166）；DINO 系最优 `outputs/dino_s_full_best.pt` /
`dino_s1_best.pt`（val 0.9188/0.9216，test 裸 0.9082/0.9085）。
DINO 全链（联合权重待补）是后继第一优先级——DINO+ce×rec 尚未跑过。

## 2. 任务树（三线，每线 = 测什么 + 参数点 + 窗口观察 + 验收数）

### (a) 锚做强（$x_t$ 上限决定一切：FO 单调律）

已知：UNeXt-FO 0.8025 → DINO-FO 0.8928/0.8955（+9.0pt，008-2 验收）。
后继：

- **a1 DINO 全链 + 联合权重**：`--anchor dinov2-small` × ce0.05/rec1.0，
  30ep×400×双 seed。验收：val/test 双口径 + mirror-HD95；与 s2-UNeXt
  （0.9166/8.38）同预算对照。这是全树第一优先级（DINO 好特征 + 最优权重的
  乘积从未测过）。
- **a2 MedSAM 真 P0/Q0**：C/D 链路阻塞中（权重指针未下载 + 1024×4 在 A30
  OOM，eval_008c §2）。两选一：① 降输入 512/384 重探显存；② 放弃 SAM 线，
  以文档级结论关闭。HDC-B/C 名副其实只在此线实现；在阻塞解除前禁谈
  "P0/Q0 对齐度"。权重进容器是 human 动作，imp 只探显存。
- **a3 高分辨 decoder / 迟上采样**：现状 decoder 输出 64（256 输入 H/4），
  eval 上采样计分（005 §1.4(b) 未做）。输出提至 128（一级上采样 + refine），
  GT 对齐仍 nearest。验收：HD95 降 + Dice 掉 <1pt（掉超即停，002 O7）。
  与 §4 mask 插值口径对照（006_post_eval §4）同跑，互为留痕。
- **a4 归一化消融**：输入归一（ImageNet stats vs echo stats，008b 未跑残留）
  + loss 归一开关（`--norm`，opt_005 未转正）。单变量各一 run，与 a1 同 budget
  可比。灰度→3ch 复制方式 freeze，不动。
- **a5 切片常驻**：ES-proxy / 小腔体 / 边界带（`eval_test.py:132` 已落地）
  进每次 test 必报项；full-mean 0.91 天花板下切片才是判据（post6 §3 已证
  平滑 trade-off 靠切片分辨）。

保留参数点：`--anchor`、`--l-boundary`（a3 权重进第七维时）、归一开关、
decoder 输出分辨率。窗口观察：val HD95（主）+ first/last-frame
（DINO 首强尾弱是否复现，eval_008 观察项）+ boundary-band。

### (b) 估计器做准（$g_t$ 误差是论文 H2/H4 的实证来源）

已知：ds DINO 下真动（0.67，eval_008）但 UNeXt 下恒等/微动；
K 漂持续（0.83–0.97，换好特征也没拉回，架构问题实锤+1）；
格 3 时钟项证伪（ds 恒等摆设，旧 P4 作废）；格 4 H4 三线阴性（三 ckpt 共识）。

- **b1 ds 真间隔重验**：`stress_ds_real.py`（格 3 已有）+ `dts` 元数据。
  PNG 无 wall-clock（data_camus DEVIATION）是硬上限；两轨：① 跳帧响应
  （stride-2/3 下 $\hat{\Delta t}$ 均值抬升 ≥5% + 真头 vs 固定 1.0 保留率差）；
  ② 元数据链路（human 给时间戳 → `load_patient` 出真 dts → 同 ckpt 重测）。
  判据：任一满足则"时钟项复活"，重写格 3 结论；否则证伪维持。
- **b2 K 门回落**：VF-deep 已部分（eval_vf）；温度（`--k-temp`）/平衡
  （`--vf-balance`）备选已证伪短预算——全预算重测只做一项 exceptionally：
  ce 回调（stage-2 记录：ce0.05 推 K 到 0.97，ce 回调或 rec 探头加权是回落方向）。
  smooth 三点（0/1e-4/1e-3，归一后）在欠训区平坦，全预算单点补测即可。
- **b3 ds-lr / smooth 网格**：`--ds-lr` {5e-5, 1e-4, 2e-4} × smooth {0, 0.01}
  小网格（6 点），短筛 6ep/60 定方向 + 胜者全预算单 seed。ds 1.08 放大步长
  （stage-2 ce 低监督下）是否健康在此判定。
- **b4 γ 诊断钩子留数**：`record_gamma` + `gamma_log` + `adjacent_ratio_cv` +
  `compose_gain` 永久保留（hdc 验收既有）；`diag/gamma_cv_val` 每 K epoch
  常驻 W-event。H4 重测需新提案（格 4 阴性后，复活门槛：估计器大改后才重测，
  不例行跑）。

保留参数点：`--ds-lr`、`--vf-deep/--vf-balance/--k-temp`、
`--l-smooth`、`--l-ce/--l-rec`、S&S 无（已删）。窗口观察（核心）：
`wstep/ds_mean/ds_min`（塌零告警 `watch/ds_min=0.05`）+ `wstep/k_mean`
（O2/O3：<0.05/>0.95 持续 200 step 动作）+ `wstep/obs_n/vf_n` 能量比
（观测强 10× 则门自然偏观测，需能量平衡）+ `diag/gamma_cv_val`。

### (c) 读出做全（$z_t \to$ 解码器的三路读出）

已知：Table 2 式单 loss 排序 iou>rec>flow（opt_005）；J=3 三候选分化正常
（best 0.43/0.28/0.29，stage-2）；α≈0.12 未动（全线）；后处理主项是 largest
（−2.65≈全开，post6 打回 parent closing 误判）；TTA 双双有害（UNeXt −4.4pt /
DINO −2.9pt，post6 修正 parent DINO+1.9pt 误报）；时序平滑拿 full 换边界
（k2/k3 边界 +1.7/+2.7pt 但 full −0.13/−0.45pt + HD95 涨，不判增益）。

- **c1 HDC-B/C 对齐度**：DINO 下 dense prompt 与 region token 是否真读 $z_t$。
  探针：α 收敛值（0.12 是否动）+ query 数 {2,4,8} 单维扫（短预算）+
  region 消融（`--no-detach` 反向已在格 2，主干 ON 不动）。
  MedSAM 真 P0/Q0 到位前，对齐度只记数不判。
- **c2 J 候选质量选择有效性**：`best_j` 分化 + quality-IoU 相关性（`l_iou`
  监督下 quality 是否真排序候选）。干预：J {1,3,5} 短预算——J=1 退化为单头，
  量化"候选机制"本身的隔离增益（从未测过）。
- **c3 后处理应用口径冻结**：largest 单开定为应用口径（post6 主项结论），
  min_size {8,16,32,64} 二阶扫 + 阈值 val 择阈（006_post_eval §3，需先查
  val 齐性）→ 冻结应用口径文档。裸口径永不掺后处理/TTA/平滑。
- **c4 TTA/平滑归档**：TTA 有害、平滑 trade-off 记归档结论，不再跑；
  TTA 候选错配追问（quality-argmax 未随 logits 平均）记观察，不修。

保留参数点：query 数、J 数、`--pp-item/--pp-min-size`、`--temporal-average`
（恒等对照 k=1 常驻）、`--tta-flip`（归档）。窗口观察：`wstep/best_j*`
（O4 垄断告警 0.95）+ `wstep/alpha_c/f` + `wstep/gamma_sat`（O1 0.3）。

## 3. 初步结果表（论文对照起点，报方向不报点处已标）

| 配置 | val Dice | test Dice（裸） | test HD95 cdist/mirror | 备注 |
|---|---|---|---|---|
| UNeXt-FO | 0.8025 | 0.8078 | 40.30 / — | H1 旧分母 |
| DINO-FO | 0.8928/0.8955 | 0.8876 | 9.77 / 10.59 | P1 转正分母（+9.0pt） |
| UNeXt grid1（ON） | 0.9120/0.9100 | 0.9110 | 11.00 / 12.76 | 主干锁定 |
| UNeXt s2（ce0.05/rec1.0） | 0.9163/0.9175 | 0.9166 | 8.38 / 9.09 | 当前最优点 |
| DINO-s-full（s0/s1） | 0.9188/0.9216 | 0.9082/0.9085 | 7.32–7.65 / 7.70–8.14 | HD95 真差 −3.5 |
| 论文 DIAG 行 | .9360 | — | 5.43 | 缺口归因见下 |

缺口归因（与 .9360/5.43）：① 基座 dts 恒 1（真间隔未用）；② λ 占位史
（s2 联合部分补，smooth/boundary 未定）；③ 单次 2x 上采样（decoder 64→
计分，高分辨未做）；④ 预训练锚只到 DINO（MedSAM 阻塞）；⑤ HD95 自研 vs
统一 surface 口径差（mirror 对齐后残差）。应用口径（largest 后处理）在
UNeXt 上 +0.26pt/−2.78，DINO 侧零增益（008b 读数）——报数必须注裸/应用。

## 4. 流程纪律（测什么 / 留什么 / 看什么的总闸）

- **搜索纪律**：多组一律网格 + 双 seed，禁手调单点引用；短预算（60/6ep）
  结论加"欠训区"限定词，转正/删除 verdict 只认 30ep 全预算（SVF 程序性
  verdict先例）。最佳点独立 seed 复跑。
- **参数点保留**：`train_camus.py` 现有 20+ flags 全部保留（§1 表 + §2 各线）；
  新增 flag 必须默认 OFF 且写进 run tag；删除 flag 禁止（冻格不删代码，
  006 铁律延续）。
- **中间窗口**（002 三层 + 009 增补，imp 必记、evaluator 必查）：
  W-step：六项拆分 + total真值（007 §5 修正：`wstep/loss_total=W_total/n`，
  旧 ce+dice 和改名 `loss_seg`）+ `loss/*_ep` 独立累加器 W_ep（007 §5，
  跨步重置污染已修）+ K/γ/sat/α/grad/ds/best（§2b/c 观察阈值见 O1–O10 +
  `watch/ds_min=0.05`）。
  W-epoch：val Dice/HD95 + 时序三件套 + first/last + 切片三件。
  W-event：ckpt（含 ds_head + md5）+ `diag/gamma_cv_val` + 诊断快照
  （H4 重测前不例行跑干预/复合）。
- **eval 展板**（`build/wiki`（旧wiki/site，已目录化））：按三线重组章节（锚/估计器/读出），每线 =
  idea 句 + 冻结配置 + 结果表 + 缺口归因；为 idea 服务，不为 run 数服务。
- **作废追溯**：旧 P4 合成抖动、test 侧择阈、DINO test 反超、closing 主项、
  DINO+1.9pt TTA——五项作废声明永久有效，后继引用即打回。

## 5. R0 复现批（第一要务：先复现，再探索；§6 原清单后移）

动机：§3 表是跨世代拼出来的（DINO test 单 ckpt、s2 跨 seed、FO 对照跨脚本、
后处理档跨世代）。R0 用 §1 冻结配置 + §4 纪律，把每个格子 locked 重跑一遍，
输出 baseline v1.0 锁定表。不引入任何新 flag、新结构；纯重跑 + 离线 eval。

复现矩阵（同 split md5 `682f3d89`、裸/应用双口径、seed {0,1} 双跑处已标）：

| # | 配置 | 现状 | R0 动作 | 验收数 |
|---|---|---|---|---|
| R0-1 | UNeXt-FO 双 seed | val 0.8025（s0），test 0.8078 | s1 test 补跑 + 双 seed test 对照 | test 裸 Dice/HD95 双 seed |
| R0-2 | UNeXt grid1 ON 双 seed | val 0.9120/0.9100，test 仅 s0 | s1 test 补跑（ckpt 已有 `grid1_s1_best.pt`） | test 裸 + 切片三件双 seed |
| R0-3 | UNeXt s2（ce0.05/rec1.0）双 seed | val 0.9163/0.9175，test 仅 s1 | s0 test 补跑（ckpt 已有 `s2full_s0_best.pt`） | test 裸 + mirror 双 seed |
| R0-4 | DINO-FO 双 seed | val 0.8928（s0）/0.8955（best），test 0.8876 | s1 FO test 补跑 + val s1 确认 | FO 分母双 seed 落定 |
| R0-5 | DINO-full 双 seed | val 0.9188/0.9216，test 裸 0.9082/0.9085 | 同 ckpt 重跑复核（幂等性）+ 切片三件补齐 | 重跑差 <0.001 即锁定 |
| R0-6 | DINO × ce0.05/rec1.0（a1） | 从未跑过 | 全预算双 seed 新跑（唯一新训练） | val/test 双口径 + mirror |
| R0-7 | 后处理应用口径 | largest 主项（UNeXt），DINO 零增益 | min_size {8,16,32,64} + val 择阈（val 齐性先查），离线 | 冻结应用口径文档 v1.0 |

判定：R0 任一格重跑差 >0.5pt（同 ckpt）或 >1pt（同配置双 seed 内）→ 该格标
FLAKY，不进 v1.0 锁定表，另起追查任务。R0 全过 → 输出 `materials/baseline_v10.md`
（锁定表 + ckpt md5 + split md5 + seed），后继一切对照以 v1.0 为分母。
R0-6 是唯一新训练（a1 原第一优先级并入 R0）；R0-7 离线零训练。

## 6. 给 implementation 的落地清单（队列任务序；本地离线先行）

队列制（`DIAG-code/queue_runner.sh`，hostname-pin，一卡一串行）：
task 文件 `<host>-<name>.sh` 进 `DIAG-code/queue/pending/` 即入队，
runner 每轮 `git pull` 自取，无需 ssh。命名：`<host>-r06-dino-s2-s0.sh` 等。

0. R0-1–R0-5 + R0-7：本地离线（CPU 可跑），不占卡，不用排队，先干。
1. R0-6（a1）：DINO × ce0.05/rec1.0 全预算双 seed——2 个 task，
   队首（全树唯一新训练优先）。
2. C2/C3：GDKVM fair + DPFR-fair 复核——2~3 个 task，排 R0-6 后。
3. b3：ds-lr/smooth 小网格短筛（6 点，短预算）——排对照后。
4. b1：跳帧响应离线重测（有 `stress_ds_real.py`，零训练）；dts 元数据链路 human。
5. a3：高分辨 decoder 128（建模项单独立项，本树只登记）。
6. a2：MedSAM 显存重探（512/384）或文档关闭（二选一）。
7. 不做：SVF 重启、原型池、scan、J>5、TTA/平滑重跑。

### 7.1 舱内可跑对照（同 split / 同十帧 / 同口径，imp 直接执行）

| # | 方法 | 入口 | 现状 | R0 后动作 |
|---|---|---|---|---|
| C1 | frame-only（UNeXt-FO / DINO-FO） | `train_camus.py --frame-only` + ckpt | R0-1/R0-4 落定分母 | H1 分母，必报 |
| C2 | GDKVM | `upstream_BanditPM/configs/gdkvm_{echo,camus}_fair_*.yaml`（舱内现成） | 未跑 | 同 split/mask 对齐重跑，记 Dice/HD95 |
| C3 | DPFR-fair | 同上 `dpfr_*fair*.yaml` | test 0.9348 既有数（006 引） | 同口径复核一行，确认数有效 |
| C4 | dies/dice 退化链 | J=1、detach-OFF（格 2 已有）、smooth-OFF | 部分有数 | 补齐 Table 2 式消融行 |

公平粒度（upstream README 既有约束延续）：同 split、同可见 clip、
同 `label_valid` 监督帧、同 foreground Dice 定义、logits 对齐到目标 mask
尺寸再计分。内部时序机制可不同，测试协议与 metric 空间必须固定。

### 7.2 舱外引用对照（只引论文数，不跑；method 章 Related Work 用）

| 方法 | 出处 | 取数 | 说明 |
|---|---|---|---|
| MemSAM | CVPR 2024（Deng et al.）| 论文 CAMUS/EchoNet 行 | SAM+时空记忆，prompted 方法参照；代码链路未确认，只引数 |
| GDKVM | ICCV 2025（Wang et al.，[github](https://github.com/wangrui2025/GDKVM)）| 论文四域行 | Eq.1 式共享状态的实例，我方立论反面；数引论文，跑用舱内 C2 |
| OSA | arXiv:2603.26188（Stiefel 流形正交状态更新） | 论文行 | 流形约束一派的最近参照，与我方对角群形成"弱群 vs 强流形"对照叙事 |
| EchoNet-Dynamic 官方基线 | [echonet/dynamic](https://github.com/echonet/dynamic)（DeeplabV3 系） | 开源数 | 数据集发布方基线，Adult 域必引 |
| DyL-UNet / MSSNet-Mamba | arXiv:2509.19052 / PMID 42202178 | 论文行 | 时序一致性 + Mamba 两条近期线，只引数不断言可比 |

规则：舱外数一律标"论文引用数（协议未对齐）"，禁与舱内裸口径并表；
并表只允许 C1–C4 + R0-6（a1）同协议行。Method 章 related 按"配准派 /
解耦派 / FiLM 派 / 记忆派（GDKVM/MemSAM）"四派写，DIAG 定位见 001 §2。
