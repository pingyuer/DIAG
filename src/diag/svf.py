"""SVF diffeomorphic warp branch, flag default OFF (003 sec.3).

Why each piece (DPFR ran off the road without them):
- Triplet (max_disp truncation + smooth reg + small-var init) is REQUIRED.
  DPFR's free tanh flow without the other two is exactly what ran away on
  low-texture ultrasound edges. Any one missing -> same failure.
- 1/4 resolution velocity: v is smooth by construction (anatomy deforms
  smoothly); predicting at full res wastes memory learning high-freq noise.
  Upsample phi, not v.
- Scaling&Squaring (6 steps): exp(v) via repeated squaring; inverse is the
  SAME function with -v (zero inversion cost). Identity grid cached per
  shape (never rebuilt per forward).
- align_corners unified False across the chain (DPFR used True; do not mix).
- Promotion criterion: warp-branch flow_prompt_delta > 0 (DPFR deltas idea).
  Else DELETE, no trick stacking. Gamma path + H4 diagnostics untouched.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffeomorphicTransform(nn.Module):
    """phi = exp(v) via scaling & squaring; phi_inv = exp(-v), same function."""

    def __init__(self, steps: int = 6) -> None:
        super().__init__()
        self.steps = int(steps)
        self._grid_cache: dict[tuple, torch.Tensor] = {}

    def identity_grid(self, b: int, h: int, w: int, dev: torch.device) -> torch.Tensor:
        key = (b, h, w, str(dev))
        g = self._grid_cache.get(key)
        if g is None or g.device != dev:
            # align_corners=False: pixel CENTERS, not edges (linspace would
            # shift by half a pixel; verified 0.6 warp error without this).
            ys, xs = torch.meshgrid(
                (torch.arange(h, device=dev).float() + 0.5) / h * 2 - 1,
                (torch.arange(w, device=dev).float() + 0.5) / w * 2 - 1,
                indexing="ij",
            )
            g = torch.stack([xs, ys], dim=-1).unsqueeze(0).expand(b, -1, -1, -1)
            self._grid_cache[key] = g
        return g

    def exp(self, v: torch.Tensor) -> torch.Tensor:
        """v: (B,2,H,W) stationary velocity (normalized units) -> phi grid.

        Correct S&S on displacement fields: u = v/2^N, then N times u <- u o u
        (compose DISPLACEMENT with itself: sample u at (id+u), add). The old
        code sampled the full coordinates and doubled (delta*2-id), which
        doubles the base position each step (verified inverse maxdiff 1.6).
        """
        b, _, h, w = v.shape
        u = v.permute(0, 2, 3, 1) / (2 ** self.steps)  # small displacement
        ident = self.identity_grid(b, h, w, v.device)
        for _ in range(self.steps):
            samp = F.grid_sample(
                u.permute(0, 3, 1, 2), ident + u, mode="bilinear",
                padding_mode="border", align_corners=False).permute(0, 2, 3, 1)
            u = u + samp  # u <- u + u(id+u): self-composition of displacement
        return ident + u

    def forward(self, v: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"phi": self.exp(v), "phi_inv": self.exp(-v)}


class SVFWarpHead(nn.Module):
    """Small stationary velocity from state; flag OFF by default."""

    def __init__(
        self,
        state_dim: int,
        max_disp: float = 0.05,
        smooth_w: float = 0.01,
        enabled: bool = False,
    ) -> None:
        super().__init__()
        self.enabled = bool(enabled)
        self.max_disp = float(max_disp)
        self.smooth_w = float(smooth_w)
        self.v_head = nn.Sequential(
            nn.Conv2d(state_dim, state_dim // 2, 3, padding=1, bias=False),
            nn.GroupNorm(max(g for g in range(min(8, state_dim // 2), 0, -1)
                             if (state_dim // 2) % g == 0), state_dim // 2),
            nn.GELU(),
            nn.Conv2d(state_dim // 2, 2, 1),
        )
        # small-var init: near-zero neighborhood start
        nn.init.normal_(self.v_head[-1].weight, std=1e-4)
        nn.init.zeros_(self.v_head[-1].bias)
        self.svf = DiffeomorphicTransform()

    def velocity(self, s: torch.Tensor) -> torch.Tensor:
        """State (B,C,H,W) at 1/4 res -> truncated velocity (B,2,H,W)."""
        v = self.v_head(s)
        return v.tanh() * self.max_disp if hasattr(v, "tanh") else torch.tanh(v) * self.max_disp

    def smooth_loss(self, v: torch.Tensor) -> torch.Tensor:
        dh = (v[:, :, 1:, :] - v[:, :, :-1, :]).pow(2).mean()
        dw = (v[:, :, :, 1:] - v[:, :, :, :-1]).pow(2).mean()
        return dh + dw

    def forward(self, s: torch.Tensor, feat_hw: tuple[int, int]) -> dict[str, torch.Tensor]:
        """s: quarter-res state. Returns phi at feat_hw (upsampled), v, smooth."""
        if not self.enabled:
            b = s.shape[0]
            dev = s.device
            return {"phi": self.svf.identity_grid(b, *feat_hw, dev),
                    "v": torch.zeros(b, 2, *feat_hw, device=dev),
                    "smooth": s.new_zeros(()),
                    "delta": s.new_zeros(())}
        v = self.velocity(s)  # (B,2,h/4,w/4)
        v_full = F.interpolate(v, size=feat_hw, mode="bilinear", align_corners=False)
        phi = self.svf.exp(v_full)
        return {"phi": phi, "v": v_full, "smooth": self.smooth_loss(v) * self.smooth_w,
                "delta": v_full.detach().abs().mean()}
