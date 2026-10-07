# 队列 runner 验收（eval n-20261002-131750-0ae ← impl n-20261001-171137-7f1）

口径依据：父 spec Done = 双容器 queue runner 常驻 + echo 任务验证通；本节点 summary 加验认领不重跑、失败隔离。

## 1. 结论：打回（blocked）——跨容器重复执行，"认领不重跑"不成立

- ✅ 双 runner 常驻：`:32237`/`:31035` 均 `tmux queue` 存活，代码同版 919a755（含 runner 自带 git pull）。
- ✅ echo 验证通：`echo-32237`/`echo-31035` 在双容器均 DONE，log 内 hostname 正确（w790fxroaip5 / 7jahayf07yly 各自认领执行）。
- ✅ 失败隔离：`task-vf-norm` FAILED 进 `failed/` 后 runner 继续下一条（`task-vf-norm2` 随后 START），链路未断。
- ❌ 认领不重跑：`task-*`/`shared-*` 任务被**双容器各执行一次**——`task-vf-norm` 在两卡均 START→FAILED（同一 GroupNorm crash）；
  `task-vf-norm2` 在两卡同时 `.running`（mlflow `vf-norm-diag2` FINISHED b2484643 + RUNNING d8fde256 并存）。
  flock+mv 只在单 runner 内防重跑；pending 经 git 分发到双容器后，认领是每卡独立的，"不重跑"跨容器不成立。

## 2. 实测复核

| 断言 | 实测 |
|---|---|
| runner 常驻双卡 | ✅ tmux queue 双存活 |
| echo 双任务通 | ✅ 双 DONE，hostname 对 |
| 失败进 failed/ + 继续 | ✅ norm FAILED→norm2 START 双卡一致 |
| shared 任务单次执行 | ❌ 双卡各跑一次（norm、norm2 均复现；mlflow 双行） |
| FAILED rc 真值 | ⚠️ norm 日志 `rc=0`——stale-runner artifact（任务启动 13:21–22 UTC 早于 rc 修复 abc7d48 的 pull；现行码已 `rc=$?`，待新失败重验） |
| 本地 pending 残留 | 注：本地 `pending/` 仍有两 task 文件（git 跟踪），远端已消费进 done/failed（各容器本地态，不回传；全局不可见，记一行） |

## 3. 科学副产品（VF 探针，不作结论）

- FINISHED b2484643（vf-norm-diag2，val 0.7021）：`obs_n` 0.18–0.23 vs `vf_n` 0.34–0.56——vf>obs，
  与" O >> f(S) → K→1"假设**反向**；该 run K 0.52–0.57。单 run 短预算，只记数。
- 探针 shape 修复 + rc 修复链（6b71cb0/abc7d48/59f9eda）读验一致；失败 crash（GroupNorm 通道错配）系修 shape 前的旧码，已修。

## 4. 打回要求（implementation 侧）

1. 语义二选一并写死：shared = 广播（每卡各跑一次，显式承认）或单次（中央互斥/mlflow 占位）；训练任务一律 hostname pin。
2. 新失败一次重验 rc 真值（现行码）。
3. mlflow 同任务双行标注（FAILED 空行 + 双 diag2），验收以 FINISHED b2484643 为准。
