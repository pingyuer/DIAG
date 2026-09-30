"""DIAG training loss Eq.17.

L = L_CE + L_Dice + lr*L_rec + ls*L_smooth + lf*L_flow + li*L_iou
(all on labeled frames Omega; lambdas configurable, defaults below are
PLACEHOLDERS until the supplement values arrive -- human must supply).

Why each term works (supervision budget argument):

- CE + Dice (weight 1, always on): pixel-wise calibration + region overlap.
  CE alone collapses to background on imbalanced echo (cavity ~5% pixels);
  Dice alone is scale-invariant and ignores calibration. Standard pair.
- L_rec (state reconstruction): decode z_t back to F_tq, MSE. Forces the
  latent state to RETAIN content (not just discriminative shortcuts); without
  it the gate can drive K->0 and states drift to a constant (posterior
  collapse of the dynamics). Small weight: reconstruction competes with
  segmentation for capacity.
- L_smooth (displacement smoothing): penalize ||S_t - S_{t-1}||^2 / dt.
  Adjacent echo frames are near-identical anatomy; large state jumps between
  them must be noise, not signal. Divided by dt so the penalty is a velocity
  (comparable across dropped/jittered sampling). Causal (no future).
- L_flow (next-frame feature prediction): linear predictor S_t -> F_{t+1},
  MSE, next frame TRAINING-ONLY supervision (inference stays causal). Gives
  the vector field f_phi a direct gradient signal (dynamics learning would
  otherwise rely only on backprop through gates, which is weak). Inference
  never sees the future.
- L_iou (mask quality supervision): BCE(predicted quality u^(j), true IoU
  of candidate j). Trains the argmax selector. Without it all u drift equal
  and inference picks arbitrarily.
- Winner-takes-all: only the BEST candidate (max IoU) gets CE+Dice against
  GT; all J get quality supervision. Losers learn "my score should be low"
  instead of distorting shared features to match GT they didn't predict.
  Shared trunk still gets gradient through winner path + quality paths.

No GT inside forward modules: gt appears ONLY here as loss target.
Next-frame F_{t+1} used ONLY in L_flow (training); never fed to PCLF input.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DiagLossWeights:
    ce: float = 1.0
    dice: float = 1.0
    rec: float = 0.1
    smooth: float = 0.01
    flow: float = 0.1
    iou: float = 0.5
    note: str = field(default="PLACEHOLDER weights; supplement values pending (human)")


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Binary Dice loss on single-channel logits (B,1,H,W) vs (B,1,H,W) 0/1."""
    prob = torch.sigmoid(logits)
    num = 2 * (prob * target).sum(dim=(1, 2, 3))
    den = (prob + target).sum(dim=(1, 2, 3)).clamp(min=eps)
    return (1 - num / den).mean()


class DiagLoss(nn.Module):
    def __init__(self, weights: DiagLossWeights | None = None, state_dim: int = 96) -> None:
        super().__init__()
        self.w = weights or DiagLossWeights()
        # rec/flow linear probes (kept inside loss so forward stays GT/next-frame free)
        self.rec_probe = nn.Conv2d(state_dim, state_dim, 1)
        self.flow_probe = nn.Conv2d(state_dim, state_dim, 1)

    def forward(
        self,
        masks: torch.Tensor,      # (T,B,J,H,W) candidate logits
        quality: torch.Tensor,    # (T,B,J) quality logits
        gt: torch.Tensor,         # (T,B,1,H,W) 0/1, only on Omega
        labeled: torch.Tensor,    # (T,B) bool mask Omega
        states_f: torch.Tensor,   # (T,B,C,Hs,Ws) fine states (rec/smooth)
        feats_f: torch.Tensor,    # (T,B,C,Hs,Ws) fine obs features (rec target)
        dts: torch.Tensor | float = 1.0,  # (T-1,) intervals for smooth/flow
    ) -> dict[str, torch.Tensor]:
        t, b, j = masks.shape[:3]
        lab = labeled.bool().reshape(t * b)
        assert lab.any(), "Omega empty: no labeled frames"
        flat_masks = masks.reshape(t * b, j, *masks.shape[3:])[lab]  # (N,J,H,W)
        flat_gt = gt.reshape(t * b, 1, *gt.shape[3:])[lab].expand(-1, j, -1, -1)
        flat_q = quality.reshape(t * b, j)[lab]  # (N,J)

        with torch.no_grad():
            prob = torch.sigmoid(flat_masks)
            inter = (prob * flat_gt).sum(dim=(2, 3))
            union = (prob + flat_gt).sum(dim=(2, 3))
            iou = inter / union.clamp(min=1e-6)  # (N,J) true IoU
            best = iou.argmax(dim=1)  # (N,) WTA index

        n = flat_masks.shape[0]
        win = flat_masks[torch.arange(n), best].unsqueeze(1)  # (N,1,H,W)
        tgt = gt.reshape(t * b, 1, *gt.shape[3:])[lab]  # (N,1,H,W)
        l_ce = F.binary_cross_entropy_with_logits(win, tgt)
        l_dice = dice_loss(win, tgt)
        l_iou = F.binary_cross_entropy_with_logits(flat_q, iou)

        # rec: state decodes back to obs feature (labeled frames only).
        # states (N,C,Hs,Ws) already 4D after boolean index; Conv2d direct.
        sf = states_f.reshape(t * b, *states_f.shape[2:])[lab]
        ff = feats_f.reshape(t * b, *feats_f.shape[2:])[lab]
        l_rec = F.mse_loss(self.rec_probe(sf), ff)

        # smooth: velocity penalty ||S_t - S_{t-1}||^2 / dt (causal pairs)
        if torch.is_tensor(dts):
            dt = dts.to(dtype=sf.dtype, device=sf.device).reshape(-1)
        else:
            dt = torch.full((t - 1,), float(dts), dtype=sf.dtype, device=sf.device)
        vel = [(states_f[i] - states_f[i - 1]).pow(2).mean() / dt[i - 1].clamp(min=1e-6)
               for i in range(1, t)]
        l_smooth = torch.stack(vel).mean()

        # flow: S_t predicts F_{t+1} (training-only next-frame supervision).
        # (T-1,B,C,H,W) -> merge to 4D, probe, compare.
        tb, c, hs, ws = states_f.shape[1], states_f.shape[2], states_f.shape[3], states_f.shape[4]
        pred = self.flow_probe(states_f[:-1].reshape((t - 1) * b, c, hs, ws))
        l_flow = F.mse_loss(pred, feats_f[1:].reshape((t - 1) * b, c, hs, ws))
        total = (self.w.ce * l_ce + self.w.dice * l_dice + self.w.rec * l_rec
                 + self.w.smooth * l_smooth + self.w.flow * l_flow + self.w.iou * l_iou)
        return {"total": total, "ce": l_ce.detach(), "dice": l_dice.detach(),
                "rec": l_rec.detach(), "smooth": l_smooth.detach(),
                "flow": l_flow.detach(), "iou": l_iou.detach(),
                "best": best.detach()}
