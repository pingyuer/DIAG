"""ds head: learned step sizes from global-token differences (003 sec.1).

ds[t] = softplus(Linear(F[t+1] - F[t])), F = spatially-pooled backbone tokens.

Why this works:
- Learned clock: probe pause / dark region -> delta_F ~= 0 -> ds -> small ->
  Euler step shrinks (clock immunity). Fixed ones cannot do this.
- softplus (not exp): linear regime near 0 keeps gradients alive for small
  deltas; exp would squash small deltas to ~1 (no discrimination) or blow up.
- bias = softplus^{-1}(1) ~= 0.54: ds starts at ~1 (identity continuation of
  the current ones regime). Training only learns deviations.
- detach: ds is a step-size multiplier, not a supervision signal. Gradients
  through ds into the backbone would create second-order coupling
  (backbone -> F -> ds -> S -> loss -> backbone AND backbone -> F -> O).
  First version cuts that path; revisit only with isolation gain proof.
- clamp 1e-3: smooth loss divides by dt; ds -> 0 would explode it.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class DsHead(nn.Module):
    """Global-token difference -> per-interval step sizes (B, T-1), >= 1e-3."""

    def __init__(self, dim: int, min_dt: float = 1e-3, delta_scale: float = 8.0) -> None:
        super().__init__()
        self.metric = nn.Linear(dim, 1)
        # Pooled energy is O(0.05); scale up so weights stay O(1).
        self.delta_scale = float(delta_scale)
        nn.init.zeros_(self.metric.weight)
        # softplus(bias) = 1 -> bias = log(exp(1)-1) ~= 0.5413
        nn.init.constant_(self.metric.bias, math.log(math.exp(1.0) - 1.0))
        self.min_dt = float(min_dt)

    def tokens(self, feats: torch.Tensor) -> torch.Tensor:
        """Motion energy: |pooled[t+1] - pooled[t]|, (T,B,C,H,W) -> (B,T-1,C).

        Absolute value is load-bearing, not cosmetic: signed deltas have
        E[d]=0 over symmetric motion, which kills the linear head's gradient
        at zero init (symmetry trap, verified). Energy has E>0 always, so
        first-order signal exists from step 0. Physically: step size depends
        on SPEED, not direction.
        """
        pooled = feats.mean(dim=(3, 4)).permute(1, 0, 2)  # (B,T,C)
        return (pooled[:, 1:] - pooled[:, :-1]).abs().detach()

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """(T,B,C,H,W) -> ds (B,T-1), strictly >= min_dt."""
        d = self.tokens(feats) * self.delta_scale  # detached inside
        ds = F.softplus(self.metric(d)).squeeze(-1)  # (B,T-1)
        return ds.clamp(min=self.min_dt)
