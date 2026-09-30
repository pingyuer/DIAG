"""DIAG mask decoder with J candidates + quality selection (paper Eq.16).

{M^(j)_t, u^(j)_t}_{j=1}^J = D(F^e_t, P_t, Q^e_t); inference picks argmax u.

Why this design works:

- Dynamic classifier (not fixed head): each candidate mask is a query-conditioned
  linear readout of shared decoder features, mask_j = w_j . F_dec. Same features,
  J semantic mixtures. Fixed J heads would each own private conv stacks (Jx cost,
  divergent features); dynamic readout shares the trunk so all candidates see the
  same boundary detail and only disagree on region semantics. This is what makes
  winner-takes-all training work: loser candidates still get gradient through the
  shared trunk via the quality loss.
- J=3: echo boundary ambiguity is typically 2-3 modes (tight/loose cavity,
  papillary inclusion). SAM uses 3 for the same reason. Larger J dilutes WTA
  supervision (each candidate wins ~1/J of frames, noisy specialization).
- Quality head init zero + bias 0 -> u=0.5 (max uncertainty, sigmoid grad 0.25
  at max). BCE against true IoU then pushes each candidate's score toward its
  actual overlap. Inference argmax is test-time model selection without GT.
- Late upsampling: attention/classifier math at H/2 (cheap), single 2x
  bilinear + conv to full res, then identity-init boundary refinement
  (out = x + zero_conv(x), step-0 = bilinear). Boundary detail without
  destabilizing early training.
- Mask-token recurrence (paper's stopped-grad short-range recurrence):
  h_t = q_t + g * Lin(q_{t-1}.detach()), g zero-init scalar. Causal, cheap,
  stable: detached past cannot create BPTT-through-time blowup; zero-init
  means step-0 ignores history (pure per-frame), training dials in temporal
  context only if it helps.
- No GT anywhere in this module (prompt-free requirement). GT appears only
  as loss target in losses.py.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskTokenRecurrence(nn.Module):
    """h_t = q_t + g * Lin(stopgrad(q_{t-1})); g init 0 (identity start)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.mix = nn.Linear(dim, dim)
        nn.init.zeros_(self.mix.weight)
        nn.init.zeros_(self.mix.bias)
        self.g = nn.Parameter(torch.tensor(0.0))

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        """q: (T, B, K, D) region tokens -> temporally mixed (T, B, K, D)."""
        t = q.shape[0]
        outs = [q[0]]
        for i in range(1, t):
            outs.append(q[i] + self.g * self.mix(q[i - 1].detach()))
        return torch.stack(outs)


class CandidateDecoder(nn.Module):
    """F_dec = Refine(Up(Conv(F^e + P))). J dynamic readouts + quality."""

    def __init__(
        self,
        feat_dim: int = 96,
        dec_dim: int = 64,
        num_queries: int = 4,
        num_candidates: int = 3,
    ) -> None:
        super().__init__()
        self.j = num_candidates
        self.fuse = nn.Sequential(
            nn.Conv2d(feat_dim, dec_dim, 3, padding=1, bias=False),
            nn.GroupNorm(max(g for g in range(min(8, dec_dim), 0, -1) if dec_dim % g == 0), dec_dim),
            nn.GELU(),
        )
        self.refine = nn.Conv2d(dec_dim, dec_dim, 3, padding=1)
        nn.init.zeros_(self.refine.weight)
        nn.init.zeros_(self.refine.bias)
        # candidate embeddings: learnable semantic mixtures over region tokens
        self.cand_embed = nn.Parameter(torch.empty(num_candidates, dec_dim))
        nn.init.trunc_normal_(self.cand_embed, std=0.02)
        self.q_to_dec = nn.Linear(feat_dim, dec_dim)
        self.q_pool = nn.Linear(num_queries * dec_dim, dec_dim)
        self.q_head = nn.Sequential(
            nn.Linear(dec_dim, dec_dim), nn.GELU(),
            nn.Linear(dec_dim, num_candidates),
        )
        nn.init.zeros_(self.q_head[-1].weight)
        nn.init.zeros_(self.q_head[-1].bias)
        self.recurrence = MaskTokenRecurrence(feat_dim)

    def forward(
        self, f_e: torch.Tensor, p: torch.Tensor, q_e: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """f_e/p: (T,B,C,H,W); q_e: (T,B,K,D). Returns masks (T,B,J,Hf,Wf),
        quality logits (T,B,J), F_dec (T,B,dec,Hf,Wf)."""
        t, b = f_e.shape[:2]
        hw = f_e.shape[-2:]
        fused = (f_e + p).flatten(0, 1)  # (TB, C, H, W)
        feat = self.fuse(fused)  # (TB, dec, H, W)
        feat = F.interpolate(feat, scale_factor=2, mode="bilinear", align_corners=False)
        f_dec = feat + self.refine(feat)  # identity-init boundary refinement
        out_hw = f_dec.shape[-2:]
        f_dec_t = f_dec.view(t, b, -1, *out_hw)

        q = self.recurrence(q_e)  # (T,B,K,D) temporally mixed
        qd = self.q_to_dec(q)  # (T,B,K,dec)
        pooled = self.q_pool(qd.flatten(2))  # (T,B,dec) region summary
        w = self.cand_embed.unsqueeze(0).unsqueeze(0) + pooled.unsqueeze(2)  # (T,B,J,dec)
        fd = f_dec_t.permute(0, 1, 3, 4, 2)  # (T,B,H,W,dec)
        masks = torch.einsum("tbhwD,tbJD->tbJhw", fd, w)
        quality = self.q_head(pooled)  # (T,B,J) logits, 0-init -> 0.5 proba
        return {"masks": masks, "quality": quality, "F_dec": f_dec_t}
