"""C3-HD95补评adapter: 加载DPFR best_raw.pth, full_window推理test50, 存预测+算HD95三档.

输入 (spec): run目录 outputs/2026-10-06/07-48-44, best_raw.pth=4000步验证最优,
split md5 682f3d8978596f040e8d6cb1ef46cbc7, test50x10帧, 无GT提示, softmax fg>=0.5.
口径: 像素单位(spacing未知, 无metadata, 只报px); 空mask帧跳过HD95并计数;
raw/mirror/后处理三档单列; 不猜ED/ES/mm.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "src")
sys.path.insert(0, "DIAG-code")
sys.path.insert(0, "upstream_BanditPM")
sys.path.insert(0, "upstream_BanditPM/src")
from diag_metrics import dice_score, hd95, hd95_mirror, postprocess_binary_mask  # noqa: E402

from omegaconf import OmegaConf  # noqa: E402
from dpfr import DPFRSegmenter  # noqa: E402

SEEDS = {
    0: {"port": "32237", "run": "outputs/2026-10-06/07-48-44"},
    1: {"port": "31035", "run": "outputs/2026-10-06/07-48-44"},
}
REMOTE = "root@172.16.240.188:/root/DIAG_fresh/upstream_BanditPM"
DATA = Path("outputs/camus_test_pull")
OUT = Path(__file__).parent / "c3_hd95"


def load_split():
    import subprocess

    split = None
    for cand in (DATA / "camus_public_datasplit_20250706.json", Path("/tmp/camus_split.json")):
        if cand.exists():
            split = json.loads(cand.read_text())
            break
    if split is None:
        subprocess.run(
            "scp -o BatchMode=yes -P 32237 "
            "root@172.16.240.188:/input0/processed/camus_png256_10f/camus_public_datasplit_20250706.json "
            "/tmp/camus_split.json",
            shell=True, check=True)
        split = json.loads(Path("/tmp/camus_split.json").read_text())
    return split


def fetch_ckpt(seed: int) -> Path:
    import subprocess

    info = SEEDS[seed]
    local = OUT / f"dpfr_s{seed}_best_raw.pth"
    if not local.exists():
        OUT.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            f"scp -o BatchMode=yes -P {info['port']} {REMOTE}/{info['run']}/best_raw.pth {local}",
            shell=True, check=True)
    return local


def build_model():
    cfg = OmegaConf.load("upstream_BanditPM/configs/dpfr_camus_fair_dense10.yaml")
    base = OmegaConf.load("upstream_BanditPM/configs/dpfr_camus.yaml")
    cfg = OmegaConf.merge(base, cfg)
    model_cfg = OmegaConf.load("upstream_BanditPM/configs/model/dpfr.yaml")
    cfg = OmegaConf.merge(cfg, model_cfg)
    # prompt-free同训练: eval本就不碰GT(见model.py else分支), 显式钉死
    cfg["model"]["dpfr"]["mask_prompt_eval"]["source"] = "anchor_or_mask"
    return DPFRSegmenter(cfg["model"]).eval(), cfg


def main():
    import subprocess

    torch.manual_seed(0)
    split = load_split()
    test_ids = split["test_data"]
    print(f"test_n={len(test_ids)}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)

    for seed in (0, 1):
        ckpt_path = fetch_ckpt(seed)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        # best_raw可能是纯state_dict或dict套娃, 兼容两种
        sd = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        model, _ = build_model()
        missing, unexpected = model.load_state_dict(sd, strict=False), None
        print(f"seed{seed} load: missing={len(missing.missing_keys)} unexpected={len(missing.unexpected_keys)}",
              flush=True)
        model.eval()
        raws, mirs, posts, dices, empty = [], [], [], [], 0
        for pid in test_ids:
            d = DATA / pid
            if not (d / "img").exists():
                subprocess.run(
                    f"scp -o BatchMode=yes -q -P 32237 -r "
                    f"root@172.16.240.188:/input0/processed/camus_png256_10f/img/{pid} "
                    f"root@172.16.240.188:/input0/processed/camus_png256_10f/gt_lv/{pid} {d}/ 2>/dev/null; "
                    f"mkdir -p {d}/img {d}/gt_lv; "
                    f"scp -o BatchMode=yes -q -P 32237 "
                    f"root@172.16.240.188:/input0/processed/camus_png256_10f/img/{pid}/*.png {d}/img/; "
                    f"scp -o BatchMode=yes -q -P 32237 "
                    f"root@172.16.240.188:/input0/processed/camus_png256_10f/gt_lv/{pid}/*.png {d}/gt_lv/",
                    shell=True, check=False)
            imgs = sorted((d / "img").glob("*.png"))[:10]
            gts = sorted((d / "gt_lv").glob("*.png"))[:10]
            x = torch.stack(
                [torch.from_numpy(np.array(Image.open(f))).float().div(255) for f in imgs]
            ).unsqueeze(0).unsqueeze(2)  # (1,10,1,256,256)
            g = torch.stack(
                [torch.from_numpy(np.array(Image.open(f))).float() for f in gts]
            ).unsqueeze(0).unsqueeze(2)
            g = (g > 0.5).float()
            with torch.no_grad():
                out = model({"rgb": x})
                prob = torch.softmax(out["final_logits"], dim=2)[:, :, 1:2]  # (1,10,1,H,W)
            pred = (prob >= 0.5).float()
            (OUT / f"dpfr_s{seed}_{pid}.pt").unlink(missing_ok=True)
            torch.save({"prob": prob.squeeze(0).cpu(), "pred": pred.squeeze(0).cpu()},
                       OUT / f"dpfr_s{seed}_{pid}.pt")
            dices.append(float(dice_score(pred, g).mean()))
            pp = postprocess_binary_mask(pred)
            for i in range(10):
                if g[0, i].sum() == 0 or pred[0, i].sum() == 0:
                    empty += 1
                    continue
                raws.append(float(hd95(pred[0, i:i + 1], g[0, i:i + 1]).mean()))
                mirs.append(float(hd95_mirror(pred[0, i:i + 1], g[0, i:i + 1]).mean()))
                posts.append(float(hd95(pp[0, i:i + 1], g[0, i:i + 1]).mean()))
        res = {
            "seed": seed,
            "dice_raw": float(np.mean(dices)),
            "hd95_cdist_px": float(np.mean(raws)) if raws else None,
            "hd95_mirror_px": float(np.mean(mirs)) if mirs else None,
            "hd95_post_px": float(np.mean(posts)) if posts else None,
            "n_scored": len(raws),
            "n_empty_skipped": empty,
            "unit": "px (spacing未知, 无metadata)",
        }
        (OUT / f"dpfr_s{seed}_hd95.json").write_text(json.dumps(res, indent=2))
        print(f"seed{seed}: dice={res['dice_raw']:.4f} cdist={res['hd95_cdist_px']} "
              f"mirror={res['hd95_mirror_px']} post={res['hd95_post_px']} "
              f"scored={res['n_scored']} empty_skip={res['n_empty_skipped']}", flush=True)


if __name__ == "__main__":
    main()
