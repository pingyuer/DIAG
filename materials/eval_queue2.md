# 队列语义验收（eval n-20261003-032419-e15 ← impl n-20261002-133355-7fd）

口径依据：打回要求三项（认领语义二选一写死；训练任务 hostname pin；新失败重验 rc）。

## 1. 结论：通过（blocked 关闭；队列转可用）

- ✅ 语义写死中央互斥（90d5a10）：runner 只取 `$ME-*`；shared 广播删除；unprefixed 忽略（fail loud）。
  双容器现行码同版 3be5c03。
- ✅ failtest rc=7 真值：`:32237`（owner 匹配）START→FAILED `rc=7`，probe 回显 + 真 rc 同日志；
  `:31035`（owner 不匹配）未执行、无日志——认领互斥成立。
- ✅ 失败隔离延续：failtest 进 `failed/`，runner 存活（tmux queue 双存活）。

## 2. 实测复核

| 打回项 | 实测 |
|---|---|
| 语义二选一写死 | ✅ 中央互斥；注释载明 shared 删除理由（per-container clone 发散 + 已验证双跑） |
| 训练任务 hostname pin | ✅ failtest 以 `tahara-w790fx…-failtest.sh` pin 32237；31035 侧 pending 可见但忽略 |
| 新失败重验 rc | ✅ `rc=7`（非上次 stale `rc=0` artifact）；现行 `rc=$?` 生效 |
| 跨容器重复 | ✅ 本次零重复（31035 对异主文件零动作） |

## 3. 残留注记（非打回）

- `:31035 pending/` 堆积三异主文件（failtest/a-deep/c-temp，属 32237 的）——按语义永不执行，
  靠 git pull 清不掉（消费是本地 mv）。建议：执行侧消费后推空提交，或定期清异主 pending。记一行，后继处理。
- `:32237 pending/` 留一条对方 vf-b-balance（同理）。
- 旧 `task-*` 影子 done 文件仍在（历史 artifact，不影响现行语义）。
