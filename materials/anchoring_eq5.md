# Anchoring Eq.5 验收（eval n-20260930-140749-0dd ← impl n-20260930-135850-bc4, commit 3f50423）

口径依据：`proposals/000_diag_paper_parse.md` §1.1（Eq.5：$x_t=(F_{tf},F_{tc})$，
$F_{tc}=\mathrm{Down}(F_{tf})$）+ `proposals/001_diag_innovations.md` §3 Anchoring 清单项。
本节点只验双尺度内容锚，不涉 PCLF/HDC/指标口径。

## 1. 结论：通过（带已记录偏差，不打回）

- $F_{tf}$ / $F_{tc}$ 形状、Down 一致性、content/observe 双接口、基座偏差记录、
  mlflow run——父节点断言全部属实，本地独立重跑复现一致。
- 偏差（已在代码内记录，非新发现）：基座为 UNeXt 随机初始化 ≠ 论文要求的预训练
  segmenter。清单允许此偏差备选（001 §3："暂用 UNeXt 随机初始化时必须记为偏差"），
  且 `BACKBONE_RECORD` + 模块 docstring 均已记录，故通过而非打回。

## 2. 实测复核（2026-09-30，本地 `.venv` torch 2.14 cpu，真 clip 10 帧 patient0001）

| 断言 | 实测 |
|---|---|
| $F_{tf}=(B,96,H/2,W/2)$，256 输入 → $(B,96,128,128)$ | ✅ 独立重跑 `(10,1,96,128,128)`（逐帧 forward 后 stack） |
| $F_{tc}=\mathrm{Down}(F_{tf})=(B,96,H/4,W/4)$ → $(B,96,64,64)$ | ✅ `(10,1,96,64,64)` |
| Down 一致性 maxdiff=0.0 | ✅ 独立 `avg_pool2d` 对照 maxdiff=0.0（无参 AvgPool×2，所见即所得） |
| `content()` → $F_{tf}$；`observe()` → $(F_{tf},F_{tc})$ | ✅ `(1,96,128,128)` / `((1,96,128,128),(1,96,64,64))` |
| BCHW 非法输入报错 | ✅ 3D 输入 `ValueError: ContentAnchor expects BCHW` |
| 参数量 7625314 | ✅ 独立统计一致 |
| 远端 `:32237` `PYTHONPATH=src` 可用 | ✅ `(1,96,128,128) (1,96,64,64)` |

## 3. 基座与偏差记录核验

- `src/diag/unext_backbone.py` 与 `upstream_BanditPM/model/modules/unext/official.py`
  `diff -q` → IDENTICAL（纯 vendor，无暗改）。
- `BACKBONE_RECORD`：`name=UNeXtOfficialBackbone, pretrained=False, checkpoint=None`，
  deviation 字段明示"H1 相关结论记偏差"。与 001 §2.5/§2.6 的定位一致
  （UNeXt 作随机初始化对照/备选，非预训练内容锚）。
- 对论文结论的影响：当前锚的表达力 ≠ 预训练内容锚；后续 H1 frame-only 缺口
  （Table 2 −2.63pt）复现时须先替换预训练基座，否则缺口不可比。此约束记于此，
  不在本节点解决。

## 4. mlflow run（exp96 `diag-env-bringup`，run `smoke-anchor-eq5`）

- `run_id=8d0c9af3…`，`tags.node=anchoring-eq5, host=local-cpu`，
  `code_sha=3f9b2f96…`（== 当前 HEAD 链，与 `.code_sha` 一致）。
- `params.backbone=UNeXtOfficialBackbone, pretrained=False`；
  `metrics.down_consistency_maxdiff=0.0, n_params=7625314.0`——与本地重跑一致。
- 该 run 在 exp96 内唯一（无重复/覆盖），命名与 tag 规范。

## 5. 与论文结论的对应关系

- 本节点是 Eq.5 的结构存在性验证（形状/Down/接口），不产生 Table 1–2 对照数值；
  H1 验收（frame-only −2.63pt）与 H3 验收均需等待 PCLF/HDC/decoder 就绪后，
  在预训练基座到位的前提下方可解释。
- 后续 evaluation 口径仍以 000 文档 §3–§4（表1 四数据集 Dice/HD95/mDice；
  表2 CAMUS ED/ES/Mean 全配置 .9427/.9218/.9317）为准，本节点无口径变更。
