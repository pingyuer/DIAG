# lint验收（eval n-20261004-130713-b28 ← impl n-20261004-125240-071）

口径依据：parent spec（ruff + basedpyright basic双过；范围src/diag/ + DIAG-code/；不改前向语义、不改训练超参）。

基线=2d44222零漂移（`git diff --stat`空）。

## 1. 双过复核（本地独立重跑）

- `ruff check src/diag/ DIAG-code/`：**All checks passed** ✅。
- `basedpyright`：**0 errors, 0 warnings, 0 notes** ✅（`pyrightconfig.json` basic档，parent新增）。

## 2. 语义无损核查（diff读验，28 files +151/−81）

- 主体是import重排/未用变量下划线化（`_tb/_hw/_batch/_svf_head`）/类型标注（`_CACHE: dict[str, tuple[...]]`）/bool().item()——化妆品级 ✅。
- 两处行为相关改动，均安全：
  - `eval_test.py`：`subprocess shell=True`加`check=False`（防scp失败崩eval）+ `band_d/pid_idx/patient_average/BACKBONE_RECORD`未用删除 + `P_all/G_all` assert非空——eval裸口径回归见§3。
  - `pclf.py`：`self.balance and _obs_norm is not None`守卫（balance默认关，冻格分支，无行为变化）+ `dts_t` assert——PCLF前向恒等。
  - `losses.py`：`run_mean==1`加`.item()`（bool()去警告，语义同）。
- upstream_BanditPM未碰 ✅（diff无此径）。

## 3. 回归：eval裸口径e2e通

- `DIAG_CKPT=outputs/grid1_s0_best.pt eval_test --anchor unext`：frame-mean 0.9110 / patient-avg 0.9110 / HD95 11.00 / split 682f3d89——与eval_008c裸表逐值一致 ✅。
- 未跑训练回归（spec禁改超参 + diff无训练超参数变动；CPU smoke可另起，不属本节点Done）。

## 4. 与论文/H结论的对应关系

- 无（工程卫生项，不进链条）。
