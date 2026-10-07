# 码仓收敛验收（eval n-20261001-084909-043 ← impl n-20261001-084508-109, 0c04421）

## 1. 结论：通过（带两条记录）

- 包入口：`src/diag/__init__.py` 在仓（TRACKED），本地 `import diag` ✅，双容器 `/root/DIAG_fresh`（均 0c04421）`import diag` + `DsHead` smoke ✅（ds (1,3)，与 anchor 一致）。
- `from diag.*` 装配：train/eval/stress/pilot/overfit 全经 `from diag.` ✅；`sys.path.insert src+DIAG-code` 双保险仍在。
- 记录 A（README 超前）："mlflow 打标 `git rev-parse --short HEAD`"——实测 `train_camus.py:242` 仍读 `/root/DIAG/.code_sha` 打 `code_sha` tag，无 `git_sha`。打标口径切换待后继（与 GitHub 验收记录 C 一致），README 先行。
- 记录 B（clone 包验证限容器）：`/tmp/diag_clone_check` 本地无 torch，包 import 未在该 clone 上验证；但双容器 fresh clone（同 commit）验证通过，结论等效。
- deploy key 双容器 `/root/.ssh/diag_github_deploy` ✅；旧 `/root/DIAG` tarball 制与 `_fresh` 并存（anchor 已注下次训练切换）。
