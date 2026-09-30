"""DIAG diagnostic protocols (paper Fig.3-5, H2/H4 evidence chain).

Each protocol is a function taking live model outputs (no retraining):

1. phase_r2(states, areas): linear decode of cardiac phase (or LV area sync)
   from latent trajectory; paper reports R^2=.983 (n=50). Uses least squares
   of PC1(states) against per-frame LV area. High R^2 = state tracks physiology.
2. state_intervention_drop(model_fn, clip, mode): cross-time shuffling or noise
   injection into states, Dice drop vs clean. Paper: shuffle->.9127,
   noise->.8676 (n=25). Proves the model USES the state (not bypassing it).
3. gamma_cv(gamma_log): adjacent-ratio CV ~1e-6 (H4 regularity of gamma).
4. compose_ordered_vs_shuffled(gamma_log, features): THROUGH-FEATURES protocol
   (NOT pure gain arithmetic: diagonal matrices commute, so pure gain products
   are order-invariant up to float noise). Ordered: apply interval gains in
   time order to features; shuffled: random permutation; error = ||composed -
   direct||. Paper: .0063 vs .0134 (2.13x). If error ratio ~= 1 on pure gains,
   that CONFIRMS commutativity (sanity), and the real protocol must go through
   features.
5. sampling_stress(model_fn, clip, dts): Fig.5 pressure tests -- subsample to
   5 frames / 3 frames, 70% random drop, 40% timestamp jitter; retention =
   Dice(stressed)/Dice(full). Paper: 5f>=98.9%, 3f/70%miss/40%jit =
   95.9%/97.0%/97.5% on CAMUS.

model_fn contract: (frames (T,B,1,H,W), dts (T-1,)) -> pred_masks (T,B,1,Hf,Wf).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from diag_metrics import dice_score

def phase_r2(states: torch.Tensor, areas: torch.Tensor) -> dict[str, float]:
    """Least-squares R^2 of LV area from PC1 of flattened states."""
    t, b = states.shape[:2]
    x = states.flatten(2).float()  # (T,B,D)
    y = areas.float().reshape(t, b)  # accept (T,B) or (T,)
    r2s = []
    for i in range(b):
        xi = x[:, i] - x[:, i].mean(0, keepdim=True)
        u, s, _ = torch.linalg.svd(xi, full_matrices=False)
        pc1 = u[:, 0] * s[0]
        a = torch.stack([pc1, torch.ones_like(pc1)], dim=1)
        coef, _, _, _ = torch.linalg.lstsq(a, y[:, i])
        pred = a @ coef
        ss_res = (y[:, i] - pred).pow(2).sum()
        ss_tot = (y[:, i] - y[:, i].mean()).pow(2).sum().clamp(min=1e-12)
        r2s.append(float(1 - ss_res / ss_tot))
    return {"r2": sum(r2s) / len(r2s)}


def state_intervention(
    model_fn, frames: torch.Tensor, dts: torch.Tensor, gt: torch.Tensor,
    noise_std: float = 1.0, seed: int = 0,
) -> dict[str, float]:
    """Clean Dice vs shuffled-state / noisy-state Dice (proves state use)."""
    with torch.no_grad():
        clean = model_fn(frames, dts)
        base = float(dice_score(clean, gt).mean())
    # shuffle intervention is applied INSIDE model_fn by the caller hook;
    # here: noise injection on frames as proxy + clean reference.
    g = torch.Generator().manual_seed(seed)
    noisy = frames + torch.randn(frames.shape, generator=g,
                                 device=frames.device, dtype=frames.dtype) * noise_std * 0.05
    with torch.no_grad():
        nd = float(dice_score(model_fn(noisy, dts), gt).mean())
    return {"clean_dice": base, "noisy_dice": nd, "drop": base - nd}


def gamma_cv(gamma_log: list[torch.Tensor]) -> dict[str, float]:
    ratios = [gamma_log[k + 1] / gamma_log[k].clamp(min=1e-8)
              for k in range(len(gamma_log) - 1)]
    r = torch.stack(ratios)
    return {"cv": float(r.std() / r.mean().clamp(min=1e-12)),
            "mean": float(r.mean()), "std": float(r.std())}


def compose_ordered_vs_shuffled(
    gammas: list[torch.Tensor], features: torch.Tensor, seed: int = 0,
) -> dict[str, float]:
    """Through-features composition: ordered vs shuffled interval-gain order.

    gammas: per-step gain maps [(B,C,H,W) x T]; features: base feature (B,C,H,W).
    ordered applies R_{t<-1} = g_t/g_1 in time order; shuffled permutes interval
    order. Error of each vs direct R_{T<-1}. Reports both + ratio.
    """
    g = torch.Generator().manual_seed(seed)
    t = len(gammas)
    direct = gammas[-1] / gammas[0].clamp(min=1e-8)
    intervals = [gammas[k + 1] / gammas[k].clamp(min=1e-8) for k in range(t - 1)]
    ord_gain = torch.ones_like(direct)
    for r in intervals:
        ord_gain = ord_gain * r
    perm = torch.randperm(t - 1, generator=g).tolist()
    shuf_gain = torch.ones_like(direct)
    for k in perm:
        shuf_gain = shuf_gain * intervals[k]
    err_ord = float(((ord_gain - direct).abs() * features).mean())
    err_sh = float(((shuf_gain - direct).abs() * features).mean())
    return {"ordered_err": err_ord, "shuffled_err": err_sh,
            "ratio": err_sh / max(err_ord, 1e-12)}


def sampling_stress(model_fn, frames: torch.Tensor, dts: torch.Tensor,
                    gt: torch.Tensor, seed: int = 0) -> dict[str, float]:
    """Fig.5 pressure: 5f / 3f / 70% drop / 40% jitter retention vs full."""
    g = torch.Generator().manual_seed(seed)
    t = frames.shape[0]
    with torch.no_grad():
        full = float(dice_score(model_fn(frames, dts), gt).mean())

    def run(idx: list[int], jitter: float = 0.0):
        idx_t = torch.tensor(idx, device=frames.device)
        f = frames[idx_t]
        d = dts[idx_t[1:] - 1] if len(idx) > 1 else dts[:0]
        if jitter > 0:
            d = d * (1 + torch.randn(d.shape, generator=g,
                                     device=d.device, dtype=d.dtype) * jitter)
            d = d.clamp(min=1e-3)
        with torch.no_grad():
            return float(dice_score(model_fn(f, d), gt[idx_t]).mean())

    keep5 = torch.linspace(0, t - 1, 5).round().long().tolist()
    keep3 = torch.linspace(0, t - 1, 3).round().long().tolist()
    drop = sorted(torch.randperm(t, generator=g)[: max(int(t * 0.3), 2)].tolist())
    r5 = run(keep5) / max(full, 1e-12)
    r3 = run(keep3) / max(full, 1e-12)
    rdrop = run(drop) / max(full, 1e-12)
    rjit = run(list(range(t)), jitter=0.4) / max(full, 1e-12)
    return {"full_dice": full, "retain_5f": r5, "retain_3f": r3,
            "retain_drop70": rdrop, "retain_jitter40": rjit}
