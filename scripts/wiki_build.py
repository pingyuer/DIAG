"""Build static wiki: index + per-node pages from materials/*.md + mlflow exp97 table."""
import re, html, os, subprocess, sys
from pathlib import Path
ROOT = Path("/home/tahara/DIAG"); MAT = ROOT/"materials"; OUT = ROOT/"build/wiki"
GROUPS = [
 ("主干与基线", ["env_bringup","anchoring_eq5","pclf_eq69","hdc_eq1015","train_eq1617","integration_001","overfit_1clip","fullchain_camus30","dualrun_h1"]),
 ("窗口与诊断", ["windows_002","metrics_diag","eval_004","hd95_align"]),
 ("重构与搜索", ["restructure_003","svf_grid","lambda_grid","opt_005"]),
 ("工程", ["repo_converge","github_push"]),
]
def md2html(md: str) -> str:
    out = []
    for line in md.split("\n"):
        if line.startswith("###### "): out.append(f"<h6>{html.escape(line[7:])}</h6>")
        elif line.startswith("##### "): out.append(f"<h5>{html.escape(line[6:])}</h5>")
        elif line.startswith("#### "): out.append(f"<h4>{html.escape(line[5:])}</h4>")
        elif line.startswith("### "): out.append(f"<h3>{html.escape(line[4:])}</h3>")
        elif line.startswith("## "): out.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif line.startswith("# "): out.append(f"<h1>{html.escape(line[2:])}</h1>")
        elif line.startswith("|"):
            cells = [html.escape(c.strip()) for c in line.strip().strip("|").split("|")]
            if all(set(c) <= set("-: ") for c in cells): continue
            out.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
        elif line.strip() == "": out.append("")
        else:
            t = html.escape(line)
            t = re.sub(r"`([^`]+)`", r"<code>\1</code>", t)
            t = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", t)
            out.append(f"<p>{t}</p>")
    s = "\n".join(out).replace("<tr>", "@@TR@@", 1)
    # wrap consecutive <tr> runs in <table> (simple pass)
    lines = s.split("\n"); res = []; in_t = False
    for ln in lines:
        if ln.startswith("<tr>") and not in_t: res.append("<table border=1 cellpadding=4>"); in_t = True
        if not ln.startswith("<tr>") and ln.strip() != "" and in_t and not ln.startswith("<"):
            pass
        if in_t and not ln.startswith("<tr>"):
            res.append("</table>"); in_t = False
        res.append(ln)
    if in_t: res.append("</table>")
    return "\n".join(res)
def mlflow_table() -> str:
    try:
        r = subprocess.run([str(ROOT/".venv/bin/python"), "-c", """
import mlflow
mlflow.set_tracking_uri("http://172.16.240.77:5000")
runs = mlflow.search_runs(["97"], order_by=["start_time DESC"])
rows = []
for _, x in runs.iterrows():
    n = x.get("tags.mlflow.runName", "?")
    try: vd = round(float(x.get("metrics.val_dice", float("nan"))), 4)
    except: vd = "—"
    try: hd = round(float(x.get("metrics.val_hd95", float("nan"))), 1)
    except: hd = "—"
    rows.append(f"<tr><td>{n}</td><td>{vd}</td><td>{hd}</td><td>{x.get('params.train_n','—')}/{x.get('params.epochs','—')}/{x.get('params.seed','—')}</td><td>{str(x.get('tags.code_sha','?'))[:8]}</td></tr>")
print("\\n".join(rows[:70]))
"""], capture_output=True, text=True, timeout=120, cwd=str(ROOT))
        rows = r.stdout.strip()
    except Exception as e:
        rows = f"<tr><td colspan=5>mlflow unreachable: {html.escape(str(e))}</td></tr>"
    return f"<table border=1 cellpadding=4><tr><th>run</th><th>val_dice</th><th>val_hd95</th><th>n/ep/seed</th><th>sha</th></tr>{rows}</table>"
OUT.mkdir(parents=True, exist_ok=True)
pages = {}
for g, names in GROUPS:
    for n in names:
        f = MAT/f"{n}.md"
        if f.exists(): pages[n] = (g, f.read_text())
        else: pages[n] = (g, f"# {n}\n\n（验收文件缺失）")
for n, (g, md) in pages.items():
    (OUT/f"{n}.html").write_text(f"""<html><head><meta charset=utf-8><title>{n}</title>
<style>body{{font-family:sans-serif;max-width:900px;margin:auto;padding:16px}}table{{border-collapse:collapse;font-size:14px}}code{{background:#f0f0f0}}</style></head>
<body><a href="index.html">← 总览</a><p>分组：{g}</p>{md2html(md)}</body></html>""")
cards = ""
for g, names in GROUPS:
    items = "".join(f'<li><a href="{n}.html">{n}</a></li>' for n in names if n in pages)
    cards += f"<h2>{g}</h2><ul>{items}</ul>"
(OUT/"index.html").write_text(f"""<html><head><meta charset=utf-8><title>DIAG 实验 wiki</title>
<style>body{{font-family:sans-serif;max-width:900px;margin:auto;padding:16px}}table{{border-collapse:collapse;font-size:14px}}</style></head>
<body><h1>DIAG 实验 wiki</h1>
<p>主轴：视频分割 = 内容锚 x_t ⊕ 观察态演化 z_t；历史只经 z_t→θ_t→ρ(g_t) 进解码器。单帧捷径直达 mask 视为耦合泄漏。</p>
<p>强 baseline：DPFR-fair test 0.9348/HD95 ~6.2；我方 ds-full test 0.9119/HD95(mirror+后处理) 9.01；论文 DIAG .9360/5.43。</p>
{cards}
<h2>exp97 全 runs（mlflow 直读，最新在上）</h2>
{mlflow_table()}
<p><small>构建时间：{__import__('datetime').datetime.now().strftime('%m-%d %H:%M')}；验收源 materials/*.md；grid1 双 seed 在跑。</small></p>
</body></html>""")
print("built", len(pages), "pages ->", OUT)
