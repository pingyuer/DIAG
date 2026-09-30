"""DIAG evaluation metrics (paper Table 1/2 + temporal behavior).

Why these metrics (each catches a distinct failure mode):

- Dice (overlap): region agreement. Necessary but NOT sufficient: a mask can
  have high Dice yet ragged borders (clinically useless contours) or jitter
  across frames (temporally incoherent). Hence the rest.
- HD95 surface-based (boundary): 95th percentile of symmetric surface
  distances. Mean/max Hausdorff are outlier-dominated; HD95 trims the worst
  5% (annotation noise lives there). Surface-based = distances computed on
  boundary point sets, not over all pixels. PLACEHOLDER: unified surface
  implementation pending supplement; current version uses exact Euclidean
  distance transform on the boundary (correct math, may differ from paper's
  implementation detail -- record deviation if numbers disagree).
- Centroid drift: ||c_t - c_1|| accumulated over the clip. Catches slow
  migration (e.g. tracking sliding off the cavity). Frame-only models show
  ~3.8x accumulated drift vs DIAG (paper).
- Area roughness: mean |a_{t+1} - 2*a_t + a_{t-1}| (second difference =
  discrete acceleration of cavity area). Catches flicker: alternating
  over/under-segmentation has near-zero mean area error but huge roughness.
  Paper reports 12.5x roughness for frame-wise inference.
- Inter-frame Dice variation: mean (1 - Dice(M_t, M_{t+1})). Direct
  temporal self-consistency without GT (usable on unlabeled frames).
  Paper: 7.9x for frame-wise.
- Patient-level aggregation: metrics computed per frame then averaged per
  patient, then across patients (NOT pooled over frames: long clips would
  dominate). Matches paper's patient-level Dice gains (Fig.4).

All functions operate on torch tensors; batch dims (..., H, W).
Empty-mask convention: Dice=1 if both empty, 0 if one empty; HD95=inf if
either empty (caller filters); temporal metrics skip empty frames.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def dice_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Binary Dice per item over trailing (H, W). pred logits or 0/1, target 0/1."""
    p = (torch.sigmoid(pred) > 0.5).float() if pred.is_floating_point() else pred.float()
    t = target.float()
    inter = (p * t).sum(dim=(-2, -1))
    denom = (p + t).sum(dim=(-2, -1))
    both_empty = denom == 0
    dice = torch.where(both_empty, torch.ones_like(inter),
                       2 * inter / denom.clamp(min=eps))
    return torch.where(both_empty & ((p - t).abs().sum(dim=(-2, -1)) > 0),
                       torch.zeros_like(dice), dice)


def _boundary(mask: torch.Tensor) -> torch.Tensor:
    """Binary boundary via erosion residual (4-conn). (..., H, W) -> same."""
    m = mask.float()
    pad = F.pad(m, (1, 1, 1, 1), value=0)
    eroded = (pad[..., 1:-1, :-2] + pad[..., 1:-1, 2:] + pad[..., :-2, 1:-1] + pad[..., 2:, 1:-1]) == 4
    return (m > 0.5) & ~eroded



def _surface_coords(mask: torch.Tensor) -> torch.Tensor:
    """Boundary pixel coords (N, 2) in (y, x)."""
    ys, xs = torch.where(_boundary(mask))
    return torch.stack([ys, xs], dim=1).float()


def hd95(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Symmetric HD95 over trailing (H, W); scalar per item. inf if either empty.

    Exact pairwise distances on boundary point sets (no approximation).
    Boundary points are few hundred at 256px, cdist is cheap and exact.
    """
    p, t = pred.float(), target.float()
    *batch, h, w = p.shape
    p = (p > 0.5)
    t = (t > 0.5)
    pf = p.reshape(-1, h, w)
    tf = t.reshape(-1, h, w)
    vals = []
    for i in range(pf.shape[0]):
        cb = _surface_coords(pf[i])
        ct = _surface_coords(tf[i])
        if cb.shape[0] == 0 or ct.shape[0] == 0:
            vals.append(float("inf"))
            continue
        d = torch.cdist(cb, ct)  # (Np, Nt)
        d1 = d.min(dim=1).values
        d2 = d.min(dim=0).values
        all_d = torch.cat([d1, d2])
        k = max(int(0.95 * (all_d.numel() - 1)), 0)
        vals.append(float(all_d.kthvalue(k + 1).values))
    return torch.tensor(vals, dtype=torch.float32, device=pred.device).view(*batch)


def centroid(mask: torch.Tensor) -> torch.Tensor:
    """Center of mass (..., 2) in (y, x); NaN if empty (caller filters)."""
    m = mask.float()
    *batch, h, w = m.shape
    ys = torch.arange(h, device=m.device, dtype=m.dtype)
    xs = torch.arange(w, device=m.device, dtype=m.dtype)
    area = m.sum(dim=(-2, -1))
    cy = (m * ys.view(1, -1, 1) if m.dim() == 3 else
          (m * ys.view(*([1] * (m.dim() - 2)), -1, 1))).sum(dim=(-2, -1)) / area.clamp(min=1e-8)
    cx = (m * xs.view(1, 1, -1) if m.dim() == 3 else
          (m * xs.view(*([1] * (m.dim() - 2)), 1, -1))).sum(dim=(-2, -1)) / area.clamp(min=1e-8)
    out = torch.stack([cy, cx], dim=-1)
    return torch.where((area > 0).unsqueeze(-1), out,
                       torch.full_like(out, float("nan")))


def area(mask: torch.Tensor) -> torch.Tensor:
    return mask.float().sum(dim=(-2, -1))


def centroid_drift(masks: torch.Tensor) -> torch.Tensor:
    """Accumulated ||c_t - c_1|| over frames. masks: (T, ..., H, W) 0/1."""
    c = centroid(masks)  # (T, ..., 2)
    d = (c - c[:1]).pow(2).sum(-1).sqrt()  # (T, ...)
    return d.nansum(dim=0)


def area_roughness(masks: torch.Tensor) -> torch.Tensor:
    """Mean |a_{t+1} - 2 a_t + a_{t-1}| (discrete acceleration of area)."""
    a = area(masks)  # (T, ...)
    if a.shape[0] < 3:
        return torch.zeros_like(a[0])
    acc = (a[2:] - 2 * a[1:-1] + a[:-2]).abs()
    return acc.mean(dim=0)


def interframe_dice_variation(masks: torch.Tensor) -> torch.Tensor:
    """Mean (1 - Dice(M_t, M_{t+1})); GT-free temporal consistency."""
    t = masks.shape[0]
    if t < 2:
        return torch.zeros(masks.shape[1:-2], device=masks.device)
    d = dice_score(masks[:-1].float(), masks[1:].float())
    return (1 - d).mean(dim=0)


def patient_average(per_frame: torch.Tensor, patient_id: torch.Tensor) -> torch.Tensor:
    """Mean over frames within each patient, then mean over patients."""
    ids = patient_id.reshape(-1)
    vals = per_frame.reshape(-1)
    out = []
    for pid in ids.unique():
        out.append(vals[ids == pid].mean())
    return torch.stack(out).mean()
