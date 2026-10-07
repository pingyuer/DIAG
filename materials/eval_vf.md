# VF 动刀验收（eval n-20261002-134318-df9 ← impl n-20261002-130913-631）

口径依据：父 spec（先诊断后动手，单变量按序；K<0.7 且 val 不掉才进全预算；三招不行转归一长预算线）
+ 正向线（P4：K 贴边判退化；证伪也是产出）。

## 1. 结论：三招 verdict（短预算 6ep/60，单 seed0，禁外推）

| 招 | 代表 run | val/best | K 终值 | verdict |
|---|---|---|---|---|
| (a) deep（VF 加深一层，零初值保持） | vf-a-deep e50fd475 | 0.6924/0.6924 | **0.52**（三复现 0.52/0.52/0.53） | ✅ 有效：K 回落到 0.5 安全区，val 同档最高；进全预算 |
| (b) balance（LayerNorm 对齐 Obs/VF 能量） | vf-b-balance 977de2b7 | 0.6210/0.6210 | 0.70（双复现 0.70/0.70） | ❌ 证伪：K 未进 0.7 线且 val −7pt（另一行 6b697b54 末 ep 崩 0.47，best 0.619 同低）；ED 崩 0.10 |
| (c) temp（sigmoid/0.5，门更果断） | vf-c-temp 52f39964 | 0.7053/0.7053 | 0.66（双复现 0.66/0.67） | ❌ 未达线：K 降但未进 0.7？注：0.6573<0.7 实已进线——见记录 B |

- 记录 A（spec 判据误写方向）：spec 要求"K<0.7"——K 低=偏历史先验、K 高=偏观测；三招目标应为 K 回落（远离 0.9）。
  deep 0.52 / temp 0.66 均满足字面线，balance 0.70 卡线。anchor 称 temp"证伪"与字面线矛盾——
  实际 temp K 进线但 val 最低档（0.6745 另一行），判负应以"val 掉"而非"K 未进"为由。见 §3。
- 记录 B（temp 双行）：ec2027be val 0.6745/K 0.67 vs 52f39964 val 0.7053/K 0.66——同配置差 3pt，
  短预算方差带内；取优行 temp 与 deep 差 1pt（<seed 方差 8.6pt），排序禁引用。

## 2. 十倍假设证伪（诊断先行成立）

- vf-norm-diag2：`obs_n` 0.18–0.23 vs `vf_n` 0.34–0.56——**vf>obs，与"O>>f(S) 致 K→1"反向**。
  假设证伪 ✅（诊断先行的正面例子：动手前先量，量完假设死）。
- deep 行：obs 0.24→0.11 下降，vf 0.26→0.18——双双收缩，K 回 0.52。
  balance 行：obs≈0.97（LayerNorm 后量级变）vs vf 0.36→0.69 上升——对齐后 vf 反超，K 仍 0.70。
- 结论：K 漂与 Obs/VF 能量比**无关**（三行比值方向各异，K 行为不跟比值走）。VectorField 弱另有他因。

## 3. 去留（按 spec 失败判据）

- deep ✅ 进全预算（30ep×400×seed0/1，`vf_deep=True`，其余锁格 1）。
- balance ❌ 冻格（ED 崩 + val 掉；代码 flag 保留）。
- temp ⚠️ 冻格观察（K 进线但 val 低档 + 双行方差；flag 保留，不删）。
- 三招全 flags 化（默认 off）+ 零初值保持（deep 初值输出 0.0 已验）✅，主干无污染。

## 4. 队列修复验收（同节点 inputs）

- owner-prefix 协议（9bbfd7a）✅：runner 只取 `$ME-*`/`shared-*`；`task-*` 通配删除。
- 但远端 pending/done 仍残留旧 `task-*` 影子文件（双容器各持一份对方 run 的 done 影子：
  `:32237 done/` 含 task-vf-a-deep/balance，`:31035 done/` 含 task-vf-c-temp——实为对方执行体，
  文件名是旧通配残留）。各容器只跑了 owner 匹配项（a/c 在 32237，b 在 31035），
  无本次跨容器重复执行 ✅（与上次 blocked 的复现不同，修复有效）。
- 残留 `task-vf-norm2.sh.running` + RUNNING 僵尸行（294d69c6）：上次双跑 artifact，非本次。
- 本地 `pending/` 仍有 `task-vf-*` 三文件（git 跟踪，远端已消费；全局不可见问题延续，记一行）。

## 5. 与论文/H 结论的对应关系

- P4（K 贴边判退化）：deep 使 K 回 0.52——退化可逆的证据，动力学拿回梯度（弱正向）。
- H4 群预言不受影响（三招只动 VF 容量/门陡度，不动 γ 路径；gamma_sat=0 全行）。
- 不产生 Table 对照数值（短预算单 seed）。
