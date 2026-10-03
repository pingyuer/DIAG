"""DIAG HDC hierarchical dynamic conditioning (paper Eq.10-15).

Theta_t = H(z_t) = (gamma_t, beta_t, D_tc, D_tf, T_tc, T_tf), three readouts
of the SAME latent state z_t = (S_tc, S_tf). All readouts are per-frame
(no temporal mixing; all dynamics live in PCLF). Modulation happens BEFORE
any decoder. No logits-level fusion anywhere (architectural requirement).

Why each branch works (design notes, not just formulas):

A. Feature affine gating (Eq.11-12): F^e = gamma * F^base + beta,
   gamma = sigmoid(G_gamma(S^f)) > 0.
   - Positivity preserves the SIGN of pretrained content channels. FiLM-style
     (1+scale) can cross zero, flipping channel semantics and breaking the
     diagonal-group closure/invertibility (Eq.4). Positive-diagonal matrices
     compose (product stays positive) and invert; sign-flipping scales do not.
   - Sigmoid zero-region has max gradient (sigma'=0.25), same reason LSTM/GRU
     gates train well from zero-init. Gate generator starts at max-sensitivity.
   - Init: G weights zero, gamma bias +2.0 -> gamma ~= 0.88 (near pass-through,
     still sigma'=0.105 healthy grad). Near-identity start means step-0 equals
     the pretrained frame-only behavior; training only learns improvements
     (residual principle). Beta zero-init -> 0.
   - H4 hook: every forward can archive detached gamma for CV / composition
     diagnostics (adjacent-ratio CV, ordered-vs-shuffled compose error).

B. Dense spatial prompt (Eq.13): P_t = P_0 + Up(D_tc) + D_tf.
   - Additive (not multiplicative): prompts inject a spatial prior ("where to
     look"), features gating selects ("what matters"). Additive zero-init
     branches keep activation variance unchanged; random multiplicative init
     would amplify variance. Hence both dense heads zero-init -> P_t = P_0
     exactly at start.
   - Coarse branch gives global region context (upsampled), fine branch local
     detail. P_0 interface accepts an external pretrained prompt map; default
     is a learnable zeros map (no pretrained segmenter available yet).

C. Region semantic tokens (Eq.14-15): Q^e = Q_0 + a_c*T_tc + a_f*T_tf.
   - Per-scale K learnable queries attend over flattened state positions:
     A = softmax(Q K^T/sqrt(D)). Each query learns one anatomical region
     prototype (blood pool / wall / background); the token is a region-pooled
     state descriptor. Separate K/V projections (attention weights vs content
     must not share a projection).
   - alphas = sigmoid(raw), raw init -2.0 -> ~=0.12 (same near-zero-start
     trick as DPFR gate_init; Q^e ~= Q_0 at start). Q_0 learnable,
     trunc-normal 0.02 (same init language as DPFR cls_token).
   - K=4 default: LV cavity, myocardium, background + one spare; attention
     cost is K*HW (4*16K fine), negligible.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class AffineGate(nn.Module):
    """HDC-A: gamma = sigmoid(G_gamma(S^f)) > 0, F^e = gamma*F^base + beta."""

    def __init__(self, state_dim: int, feat_dim: int, gamma_bias_init: float = 2.0) -> None:
        super().__init__()
        self.g_gamma = nn.Conv2d(state_dim, feat_dim, 1)
        self.g_beta = nn.Conv2d(state_dim, feat_dim, 1)
        nn.init.zeros_(self.g_gamma.weight)
        nn.init.constant_(self.g_gamma.bias, float(gamma_bias_init))
        nn.init.zeros_(self.g_beta.weight)
        nn.init.zeros_(self.g_beta.bias)

    def forward(
        self, s_f: torch.Tensor, f_base: torch.Tensor, archive: list | None = None
    ) -> dict[str, torch.Tensor]:
        """s_f, f_base: (B, C, H, W) single frame (or T*B merged)."""
        hw = f_base.shape[-2:]
        gamma = torch.sigmoid(self.g_gamma(s_f))
        beta = self.g_beta(s_f)
        if gamma.shape[-2:] != hw:
            gamma = F.interpolate(gamma, size=hw, mode="bilinear", align_corners=False)
            beta = F.interpolate(beta, size=hw, mode="bilinear", align_corners=False)
        f_e = gamma * f_base + beta
        if archive is not None:
            archive.append(gamma.detach())
        return {"F_e": f_e, "gamma": gamma, "beta": beta}


class DensePromptHead(nn.Module):
    """HDC-B: P_t = P_0 + Up(D_tc) + D_tf. Heads zero-init -> identity start."""

    def __init__(
        self,
        state_dim_fine: int,
        state_dim_coarse: int,
        prompt_dim: int,
        prompt_hw: tuple[int, int] | None = None,
    ) -> None:
        super().__init__()
        self.head_f = nn.Conv2d(state_dim_fine, prompt_dim, 1)
        self.head_c = nn.Conv2d(state_dim_coarse, prompt_dim, 1)
        nn.init.zeros_(self.head_f.weight)
        nn.init.zeros_(self.head_f.bias)
        nn.init.zeros_(self.head_c.weight)
        nn.init.zeros_(self.head_c.bias)
        if prompt_hw is not None:
            self.p0 = nn.Parameter(torch.zeros(1, prompt_dim, *prompt_hw))
        else:
            self.p0 = None

    def forward(
        self,
        s_tf: torch.Tensor,
        s_tc: torch.Tensor,
        out_hw: tuple[int, int],
        p0: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        d_f = self.head_f(s_tf)
        d_c = self.head_c(s_tc)
        if d_f.shape[-2:] != out_hw:
            d_f = F.interpolate(d_f, size=out_hw, mode="bilinear", align_corners=False)
        d_c_up = F.interpolate(d_c, size=out_hw, mode="bilinear", align_corners=False)
        base = p0 if p0 is not None else (
            self.p0.expand(s_tf.shape[0], -1, -1, -1)
            if self.p0 is not None and tuple(self.p0.shape[-2:]) == tuple(out_hw)
            else torch.zeros_like(d_f)
        )
        return {"P": base + d_c_up + d_f, "D_tf": d_f, "D_tc_up": d_c_up}


class RegionTokenHead(nn.Module):
    """HDC-C: K queries attend to flattened state; Q^e = Q_0 + a_c*T_c + a_f*T_f."""

    def __init__(self, state_dim: int, num_queries: int = 4, alpha_init: float = -2.0,
                 detach_s: bool = True) -> None:
        super().__init__()
        self.dim = state_dim
        # 006 grid2: detach on/off ablation switch. True = one-sided detach
        # (S gets no region-path gradients); False = full backprop into S.
        self.detach_s = bool(detach_s)
        self.queries = nn.Parameter(torch.empty(num_queries, state_dim))
        nn.init.trunc_normal_(self.queries, std=0.02)
        self.k_proj = nn.Linear(state_dim, state_dim)
        self.v_proj = nn.Linear(state_dim, state_dim)
        self.pi = nn.Linear(state_dim, state_dim)

    def tokens(self, s: torch.Tensor) -> torch.Tensor:
        """s: (B, C, H, W) -> region tokens (B, K, C)."""
        b, c, h, w = s.shape
        seq = s.flatten(2).transpose(1, 2)  # (B, HW, C)
        # One-sided detach (003 sec.2): direct loss path into S through
        # k/v would drag the dynamics state toward single-frame shortcuts.
        # Queries keep gradients (they must learn region prototypes); S does
        # not receive region-path gradients (it learns from gates + rec/flow).
        seq_sg = seq.detach() if self.detach_s else seq
        k = self.k_proj(seq_sg)
        v = self.v_proj(seq_sg)
        q = self.queries.unsqueeze(0).expand(b, -1, -1)  # (B, K, C)
        attn = torch.softmax(q @ k.transpose(1, 2) / math.sqrt(self.dim), dim=-1)
        return self.pi(attn @ v)  # (B, K, C)


class HDC(nn.Module):
    """Full HDC: theta_t = H(z_t). Frame-wise over (T, B, ...) sequences."""

    def __init__(
        self,
        state_dim_fine: int,
        state_dim_coarse: int | None = None,
        feat_dim: int | None = None,
        prompt_dim: int | None = None,
        prompt_hw: tuple[int, int] | None = None,
        num_queries: int = 4,
        gamma_bias_init: float = 2.0,
        alpha_init: float = -2.0,
        detach_region_s: bool = True,
    ) -> None:
        super().__init__()
        coarse = state_dim_coarse or state_dim_fine
        feat = feat_dim or state_dim_fine
        self.gate = AffineGate(state_dim_fine, feat, gamma_bias_init)
        self.prompt = DensePromptHead(state_dim_fine, coarse, prompt_dim or feat, prompt_hw)
        self.detach_region_s = bool(detach_region_s)
        self.region_f = RegionTokenHead(state_dim_fine, num_queries,
                                        detach_s=self.detach_region_s)
        self.region_c = RegionTokenHead(coarse, num_queries,
                                        detach_s=self.detach_region_s)
        token_dim = state_dim_fine
        assert coarse == state_dim_fine, (
            "region tokens require matching fine/coarse dims for shared Q_0; "
            f"got {state_dim_fine} vs {coarse} (project coarse first if they differ)"
        )
        self.q0 = nn.Parameter(torch.empty(num_queries, token_dim))
        nn.init.trunc_normal_(self.q0, std=0.02)
        self.raw_alpha_c = nn.Parameter(torch.tensor(float(alpha_init)))
        self.raw_alpha_f = nn.Parameter(torch.tensor(float(alpha_init)))

    @property
    def alphas(self) -> tuple[float, float]:
        with torch.no_grad():
            return (float(torch.sigmoid(self.raw_alpha_c)),
                    float(torch.sigmoid(self.raw_alpha_f)))

    def forward(
        self,
        f_base: torch.Tensor,
        s_tf: torch.Tensor,
        s_tc: torch.Tensor,
        p0: torch.Tensor | None = None,
        record_gamma: bool = False,
    ) -> dict[str, torch.Tensor]:
        """f_base/s_tf/s_tc: (T, B, C, H, W). Returns F_e, P, Q_e + intermediates."""
        t, b = f_base.shape[:2]

        def merge(x: torch.Tensor) -> torch.Tensor:
            return x.flatten(0, 1)

        archive: list | None = [] if record_gamma else None
        a = self.gate(merge(s_tf), merge(f_base), archive)
        f_e = a["F_e"].view(t, b, *a["F_e"].shape[1:])
        gamma = a["gamma"].view(t, b, *a["gamma"].shape[1:])
        beta = a["beta"].view(t, b, *a["beta"].shape[1:])

        hw = f_base.shape[-2:]
        p = self.prompt(merge(s_tf), merge(s_tc), hw, p0)
        p_t = p["P"].view(t, b, *p["P"].shape[1:])

        t_c = self.region_c.tokens(merge(s_tc)).view(t, b, -1, self.region_c.dim)
        t_f = self.region_f.tokens(merge(s_tf)).view(t, b, -1, self.region_f.dim)
        a_c = torch.sigmoid(self.raw_alpha_c)
        a_f = torch.sigmoid(self.raw_alpha_f)
        q_e = self.q0.unsqueeze(0).unsqueeze(0) + a_c * t_c + a_f * t_f  # (T,B,K,D)

        out: dict[str, torch.Tensor] = {
            "F_e": f_e, "P": p_t, "Q_e": q_e,
            "gamma": gamma, "beta": beta,
            "T_tc": t_c, "T_tf": t_f,
            "alpha_c": a_c.detach(), "alpha_f": a_f.detach(),
        }
        if archive is not None:
            out["gamma_log"] = archive  # type: ignore[assignment]
        return out


def compose_gain(gammas: list[torch.Tensor], i: int, j: int) -> torch.Tensor:
    """Interval-composed gain R_{j<-i} = gamma_j / gamma_i (diagonal group)."""
    return gammas[j] / gammas[i].clamp(min=1e-8)


def adjacent_ratio_cv(gammas: list[torch.Tensor]) -> dict[str, float]:
    """CV of adjacent-step gain ratios; paper reports ~1e-6 for trained gamma.

    NOTE: for exact diagonal math, ordered vs shuffled interval products are
    identical up to float noise (diagonal matrices commute). The paper's
    .0063-vs-.0134 gap therefore measures composed-gain prediction error
    THROUGH features/the network, not pure gain arithmetic. Use compose_gain
    applied to features for that protocol (evaluation node).
    """
    ratios = [(gammas[k + 1] / gammas[k].clamp(min=1e-8)) for k in range(len(gammas) - 1)]
    r = torch.stack(ratios)
    cv = float(r.std() / r.mean().clamp(min=1e-12))
    return {"cv": cv, "mean": float(r.mean()), "std": float(r.std())}
