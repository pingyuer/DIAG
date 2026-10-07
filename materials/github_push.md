# GitHub 首推验收（eval n-20261001-084315-7da ← impl n-20261001-084102-6d3）

## 1. 结论：有条件通过（anchor 两处与实测不符，见记录）

- 远端可达：`git ls-remote origin main` == 本地 HEAD `0c04421` ✅；`git clone --depth 5` 全新通过 ✅。
- 记录 A（anchor 不实）："clone 验证通过（有 DIAG-code/materials/proposals/src）"——实测 fresh clone 仅 `DIAG-code/README/src/pyproject/uv.lock`，**无 `materials/`、无 `proposals/*.md`**（0c04421 起 gitignored）。anchor 的"有 materials/proposals"与实测矛盾。
- 记录 B（ignore 生效 ✅）：clone 仓 `.gitignore` 逐项验证——`materials/`、`proposals/000–004`、`outputs/`、`*.pt`、`.code_sha`、`.venv`、`__pycache__/`、`.board/*.db` 全命中。本地 `git ls-files` 中 materials/proposals 计数 0。工作台件本地 13+5 文件齐（`materials/svf_grid.md` 在内），不进仓符合 0c04421 意图。
- 记录 C（PDF 例外）：`proposals/35002_DIAG_Dynamics_Induced_Af.pdf` 未被 ignore（`git ls-files` 待确认项，5 个 md 被拦但 pdf 路径 `proposals/35002...pdf` 不在 ignore 列表——`ls-files` 输出被截断，需后继确认 pdf 是否入仓；若 1.2MB pdf 入仓符合"本地备查"则无碍）。

## 2. 与 README 口径的交叉检查

- README："论文依据 `proposals/35002...pdf`（本地备查，不进仓）"——若 pdf 实际已 push 入仓，则 README 与现实矛盾。fresh clone 无 proposals 目录证明 pdf **不在**远端 main（depth 5 内），"不进仓"成立——但这是因为整个 proposals 被拦（连带 000–004 md），与"备查"表述的意图（pdf 可选、md 应在？）需 human 澄清。当前状态自洽可用，不打回。
- mlflow `git_sha` 替代 `code_sha`：待后继 ✅（当前 run 仍 code_sha，与 anchor 一致）。

## 3. 后继建议

- 若需 materials/proposals 进仓协作：改 ignore 为仅拦 `outputs/*.pt` 等产物 + pdf，白名单 md。当前全拦状态下 evaluator 验收文件只活在本地盘。
