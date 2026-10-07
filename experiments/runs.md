# runs索引（只建索引，不拉回产物）

远端产物留容器原run目录，按需scp。ckpt `*.pt` 不进仓。

- C3 DPFR s0/s1：`/root/DIAG_fresh/upstream_BanditPM/outputs/2026-10-06/07-48-44/best_raw.pth`（32237/31035各一）
- 011A/B：`diag-011` experiment（mlflow exp），run名 011a/011b-s0/s1
- GDKVM fair：exp98 `gdkvm-compare`
- 本地eval产物：`outputs/`（gitignored）
