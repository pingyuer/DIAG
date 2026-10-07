"""实验地图页：只用 materials verdict + mlflow 有效 runs，不贴 board 原文。"""
import subprocess
from pathlib import Path
ROOT = Path("/home/tahara/DIAG")
MAP = """# 实验地图：尝试 → 读数 → 追问

 > 口径：每行是一条链上的一个读数，单次不关闭问题。坏点/smoke/僵尸 run 不在此表。

 ## 主干（Eq.5–17 结构闭环）

 | 尝试 | 读数 | 追问/下一步 |
 |---|---|---|
 | Anchoring 双尺度内容锚 | F_tf/F_tc 形状对，Down 一致 maxdiff=0 | ● 结构闭合；UNeXt 随机初始化记偏差 |
 | PCLF 因果递推 | S1=O1，dt 路径活，扰动过去不变 | ● 结构闭合；门区间以重跑值为准 |
 | HDC 三分支 | γ恒正/α≈0.12/恒等启动/梯度全覆盖 | ● 结构闭合；训练态读数见 P5/P10 链 |
 | 训练/推理 J候选+六项loss | total可反传，无 GT 泄漏 | ● 结构闭合；λ 方向见 P9 链 |
 | 指标/诊断管线 | 恒等测试全过，CAMUS3 链路通 | ● 结构闭合；R2 伪相关禁引用 |

 ## P1/P4/P9 链（全预算读数累积）

 | 尝试 | 读数 | 追问/下一步 |
 |---|---|---|
 | 格1主干锁定双seed | best val 0.9120/0.9100（Δ=0.2pt），test 0.9136/8.22 | ● 方差终结；P3/P4/P5 基线锁定 |
 | 格2 detach on/off双seed | s0−0.19/s1+0.86，K照漂 | ▶ 方向分裂冻格，待新链拆增益与门控 |
 | stage-2 ce0.05×rec1.0双seed | +0.5pt双seed同向；K 0.97红灯同源 | ▶ 追问ce回调或rec探头加权 |
 | DINO全链/FO | val+0.7/HD95−3.5，test−0.5+gap；FO分母+9.0pt | ▶ 追问test分裂；MedSAM/SAM未跑 |
 | HD95 三档（mirror/后处理/sweep） | 10.64→8.29（管线 −2.3），剩余 ~3.6 是模型差距 | ● 尺子对齐完；阈值非瓶颈 |

 ## P3/P5/P10 链（读数阴性，追问中）

 | 尝试 | 读数 | 追问/下一步 |
 |---|---|---|
 | 格3真ds双轨 | max_gap 0.0004；ds静止零区分，能量淹没 | ▶ 追问能量信号根因，重设计后再读 |
 | 格4三ckpt群诊断 | CV 1e-2；ratio 1.00/drop<0.001 | ▶ 追问诊断口径与架构适配 |
 | VF三招短预算 | deep K回0.52进全预算；balance/temp冻格 | ▶ deep已进主干；其余冻格 |
 | SVF 18点网格 | val 0.65–0.73全塌，disp倒U | ❄️ 程序性摘除（6ep格内有效）；回炉待主干稳定 |
 | λ网格/005短预算 | ce降/rec升正向；smooth毒药方向；单点禁引用 | ▶ 方向先验已转入stage-2 |
 | 跨域 G2 | 0.039 零样本崩 | 🛑 gate-stop，不追全量 |

 ## 强 baseline

 DPFR-fair test 0.9348 / HD95 ~6.2（同 split 同协议）；论文 DIAG .9360/5.43。
 """
(ROOT/"docs"/"map.md").write_text(MAP)
# runs 表：链条读数行（结论行；diagnostic 同行；RUNNING/zombia 去重以 FINISHED 为准）
KEEP = ["ds-full-30ep","camus-fullchain-30ep","frameonly-30ep","grid1-lock-s0","grid1-lock-s1",
 "grid2-off-s0","grid2-off-s1","s2-full-ce005-rec10-s0","s2-full-ce005-rec10-s1",
 "dino-s-full-30ep","dino-s-full-s1","dino-fo-full",
 "dino-small-frameonly-6ep","dino-base-frameonly-6ep","sam-sam-frameonly-6ep","sam-medsam-frameonly-6ep",
 "grid3-ds-real","grid4-gamma-3ckpt","vf-a-deep","vf-b-balance","vf-c-temp",
 "ds-stress-p1p6","cardiacuda-g2-pilot","r003-smoke","overfit-1clip-ckpt",
 "L-ce-0.01","L-ce-0.01-seed7","L-rec-1.0","L-iou-1.0","L-smooth-1.0","L-dice-0.01",
 "s2-ce0.05-rec1.0-s0","s2-ce0.05-rec1.0-s1","s2-ce0.01-rec1.0-s0","s2-ce0.01-rec1.0-s1",
 "svf-g-4-0.05-0.001","center","smooth-0-6ep","off-flow-6ep","off-rec-6ep","off-iou-6ep",
 "boundary-001-6ep","ds-opt-6ep","norm-on-6ep","win-smoke2"]
r = subprocess.run([str(ROOT/".venv/bin/python"), "-c", """
import mlflow
mlflow.set_tracking_uri("http://172.16.240.77:5000")
runs = mlflow.search_runs(["97"], order_by=["start_time DESC"])
keep = """ + repr(KEEP) + """
seen = set(); lines = []
for _, x in runs.iterrows():
    n = x.get("tags.mlflow.runName", "?")
    if n not in keep or n in seen: continue
    seen.add(n)
    try: vd = round(float(x.get("metrics.val_dice", float("nan"))), 4)
    except: vd = "—"
    try: hd = round(float(x.get("metrics.val_hd95", float("nan"))), 1)
    except: hd = "—"
    lines.append(f"| {n} | {vd} | {hd} | {x.get('params.train_n','—')}/{x.get('params.epochs','—')}/{x.get('params.seed','—')} |")
print("\\n".join(lines))
"""], capture_output=True, text=True, timeout=120, cwd=str(ROOT))
(ROOT/"docs"/"runs.md").write_text("# 链条读数 runs（结论行；diagnostic 同行；坏点/smoke 已剔）\n\n| run | val_dice | val_hd95 | n/ep/seed |\n|---|---|---|---|\n" + r.stdout.strip() + "\n")
print("map+runs written")
