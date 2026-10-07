# 工作区结构整理：现状清单、目标映射、迁移步骤（唯一有效版本）

依据：用户 21:58 conductor 指令 + 实测（2026-10-06）+ 六条修正 + 追认指示。只做结构，不重训、不改算法、不覆盖历史。旧结论已就地修正，无补丁节。

## 1. 仓库边界（已核实）

| 路径 | Git 根 | remote | 状态 |
|---|---|---|---|
| `/home/tahara/DIAG`（外层） | 自身，`main` | `git@github.com:pingyuer/DIAG.git` | materials 37 份 untracked（`.gitignore:29 /materials/` 整目录忽略）；docs/ 仅 2 份跟踪；已跟踪文件零修改 |
| `upstream_BanditPM/` | 独立仓 | `https://github.com/pingyuer/BanditPM.git`，`main`，HEAD `c507adc` | 1 条本地修改 `training/logging.py`（五方法加 `*a` 透传，该文件 last-commit `945c173`，diff 见 §7），原样保护 |
| `DIAG-code/` | 无独立 `.git`，由外层跟踪 26 文件 | —— | 外层仓内目录，非独立仓库 |
| `src/diag/` | 由外层跟踪，13 模块 | —— | 唯一外部消费者 `DIAG-code/*.py`（`from diag`）；历史实现 |
| 远端 `/root/DIAG_fresh` | tarball 同步产物，非 git clone | —— | 本轮远端不动 |

## 2. 现状问题

1. `materials/` 37 份：35 与 `docs/` 同名相同，`compare_table.md` 不同，`eval_010.md` 独有。权威源=materials（保全后再去重，见 S1）。
2. `upstream_BanditPM` 14M：DPFR 锚点在 `dpfr/`；`dpfr/model.py:10` 直引 `model.modules.unext`——`model/` 是 DPFR 运行依赖（已追调用，非裁定点）。
3. `DIAG-code` 混装：方法专用脚本（历史实现）+ 工作区工具（`sync.sh`、`queue_runner.sh`、`queue/`——pending 任务证明是外层调度器，远端 runner 消费；定性为工作区工具，非裁定点）。
4. `outputs/*.pt` 不进仓；远端产物建索引不拉回。
5. `wiki/` 只有构建脚本 + `site/` 生成物；`site/` 应进构建目录，不随源码维护。

## 3. 目标映射

| 现状 | 目标 | 动作 |
|---|---|---|
| 外层根文件（README/AGENTS/pyproject/uv.lock/.gitignore 等） | 不动 | 只改边界说明 |
| `upstream_BanditPM/{dpfr,dataset,training,evaluation,configs}` + `model/modules/unext`（DPFR 直接依赖）+ logging 修改 | `repos/diag/` 内容来源 | S3 按 §8 落；本轮只登记版本 + 修改 |
| 旧 `src/diag` + `DIAG-code` 方法脚本 | `repos/diag-legacy/` | 历史实现，不占当前方法入口；S3 前原位不动 |
| `DIAG-code/{sync.sh,queue_runner.sh,queue/}` | `scripts/` | S1 移动 + 改引用 |
| `upstream_BanditPM` 其余（GDKVM 等） | `repos/baselines/<method>` 按需登记 | 本轮只登记 |
| `materials/` 37 份 | 保全提交 → 去重（S1） | 先解 ignore + `git add materials/` 提交保全；`compare_table.md` 取 materials 版（docs 版删）、`eval_010.md` 保留；35 相同者 md5 留档后删 docs 侧副本 |
| `docs/diag_project_alignment.md`、`dpfr_diag_protocol_manual.md` | `docs/project/` | 移动（历史出处注保留） |
| R1-R3/HD95三档/split md5（散在验收） | `docs/protocols/` 索引页 | 只索引原出处，不抄成"现行协议" |
| 已定事项（DINO baseline/SVF删/反超作废/DPFR即DIAG） | `docs/decisions/` | 按时间列 + 指回 board 节点；历史资料只注出处，不自动转有效协议 |
| `proposals/000-011` | `proposals/` 不动 | 补 006/007 登记 |
| 旧 PDF | `paper/references/` | 移动 |
| `wiki/site/` | 构建目录（gitignored） | 不随源码维护；`wiki/*.py`→`scripts/`，同步改 `rebuild.sh`+mkdocs+role入口+proposals链接+节点输入 |
| `.board/`、`.omp/`、`outputs/` | 不动 | `experiments/runs/` 只建索引 |

## 4. 迁移步骤

- S1：`gitignore` 去 `/materials/` → `git add materials/` 提交保全 → 按 §3 去重（docs 侧删 35+1，留档 md5）；wiki 构建目录化 + 脚本移 `scripts/` + 全引用同步（含 role/proposals/mkdocs/节点输入，旧路径注出处）；PDF → `paper/references/`。验证：`git status` 干净对应项 + `rebuild.sh` 通 + grep 全面引用零残留。
- S2：`docs/project|protocols|decisions/` + `experiments/` + `reports/` 骨架 + 首批登记；历史资料只注出处不转正。S4：charter/roles 入口 + 三回执 + 汇报。

## 5. 运行入口（S3 前不变）

本地 `uv run DIAG-code/train_camus.py`；远端 tarball 同命令。外层 `scripts/` 留旧路径 shim（S1 建）。

## 6. S3：源码历史管理 vs 外层引用（分开）

- 源码历史（独立仓内事）：推荐 B（新仓单次导入 + 附上游 commit `c507adc` 指针 + logging diff 原样带入），A（subtree split）备选，C（submodule 整仓）不取（DPFR/GDKVM 未拆）。
- 外层引用（本仓事）：先保留独立仓 + 入口整理——`repos/diag` 先为文档入口（指上游路径 + 版本 + 本地修改记录），代码搬迁等 B 执行才发生。两者解耦，入口整理不等搬迁。
- 兼容：S3 前运行路径不动；`model/modules/unext` 随 DPFR 走（调用证据确凿）；`queue` 系工作区工具（pending 任务为证）。
