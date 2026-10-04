"""DIAG PCLF predict-correct latent flow (paper Eq.6-9).

Online causal recurrence, per scale q in {c, f}:

- Eq.6 obs encode:  O_tq = O^q(F_tq)
- Eq.6 init:        S^q_1 = O^q_1
- Eq.7 prediction:  Sbar^q_t = S^q_{t-1} + dt * f^q_phi(S^q_{t-1})
  (history + true frame interval only; dynamics prior before obs arrives)
- Eq.8 correct gate: K^q_t = sigmoid(C^q[Sbar^q_t, O^q_t])
- Eq.9 correction:   S^q_t = (1-K^q_t)*Sbar^q_t + K^q_t*O^q_t

Input: (F_tf, F_tc) from ContentAnchor.observe(); output z_t = (S_tc, S_tf)
for HDC readouts. Do NOT reuse DPFRDualPromptEncoder (full-window
bidirectional, conflicts with causality).

Design:
- O^q: 1x1 Conv + GroupNorm + GELU + 1x1 Conv (channels preserved).
- f^q_phi: 3x3 Conv (pad 1, continuous-time vector field) + GN + GELU
  + 3x3 Conv, last layer zero-init so Sbar ~= S at start (near-zero start,
  same trick as DPFR gate_init).
- C^q: 1x1 Conv on concat[Sbar, O] -> sigmoid, per-channel/spatial gate.
- dt: true frame intervals from caller (scalar or (T-1,) sequence).
  dt=0 reduces prediction to Sbar=S (correction only).
"""
from __future__ import annotations

import torch
from torch import nn


def _groups(channels: int, preferred: int = 8) -> int:
    return max(g for g in range(min(preferred, channels), 0, -1) if channels % g == 0)


class ObsEncoder(nn.Module):
    """O^q: within-scale obs encoding F_tq -> O_tq (Eq.6)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.GroupNorm(_groups(channels), channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1),
        )

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        return self.net(f)


class VectorField(nn.Module):
    """f^q_phi: conv dynamics vector field S -> dS/dt (RHS of Eq.7)."""

    def __init__(self, channels: int, deep: bool = False) -> None:
        super().__init__()
        # (a) deep: extra 3x3-GN-GELU block; last layer zero-init kept
        # (identity start preserved: deeper net still outputs ~0 at init).
        if deep:
            self.net = nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                nn.GroupNorm(_groups(channels), channels),
                nn.GELU(),
                nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                nn.GroupNorm(_groups(channels), channels),
                nn.GELU(),
                nn.Conv2d(channels, channels, 3, padding=1),
            )
        else:
            self.net = nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                nn.GroupNorm(_groups(channels), channels),
                nn.GELU(),
                nn.Conv2d(channels, channels, 3, padding=1),
            )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.net(s)


class CorrectGate(nn.Module):
    """C^q: correction gate K = sigmoid(C([Sbar, O])) (Eq.8)."""

    def __init__(self, channels: int, temperature: float = 1.0) -> None:
        super().__init__()
        self.net = nn.Conv2d(2 * channels, channels, 1)
        # (c) temperature: sigmoid(x/T); T<1 sharpens the gate (more decisive).
        self.temperature = float(temperature)

    def forward(self, s_bar: torch.Tensor, o: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(torch.cat([s_bar, o], dim=1)) / self.temperature)


class ScaleFlow(nn.Module):
    """Single-scale causal recurrence: forward(feats[T], dts[T-1]) -> states[T]."""

    def __init__(self, channels: int, vf_deep: bool = False, k_temp: float = 1.0,
                 balance: bool = False) -> None:
        super().__init__()
        self.obs = ObsEncoder(channels)
        self.vf = VectorField(channels, deep=vf_deep)
        self.gate = CorrectGate(channels, temperature=k_temp)
        # (b) balance: LayerNorm aligns Obs/VF output energy so the gate
        # compares like with like instead of raw magnitudes.
        self.balance = bool(balance)
        self.obs_norm = nn.LayerNorm(channels) if balance else None
        self.vf_norm = nn.LayerNorm(channels) if balance else None

    def forward(
        self, feats: torch.Tensor, dts: torch.Tensor | float = 1.0
    ) -> dict[str, torch.Tensor]:
        """feats: (T, B, C, H, W) single-scale feature sequence.

        dts: scalar or (T-1,) true intervals. Returns states (T,B,C,H,W),
        preds (Sbar), obs, gates (K).
        """
        t = feats.shape[0]
        if torch.is_tensor(dts):
            dts = dts.to(dtype=feats.dtype, device=feats.device).reshape(-1)
            assert dts.numel() in (1, max(t - 1, 0)), (tuple(dts.shape), t)
            if dts.numel() == 1:
                dts = dts.expand(max(t - 1, 0))
        else:
            dts_t: torch.Tensor = torch.full(
                (max(t - 1, 0),), float(dts), dtype=feats.dtype, device=feats.device)
            dts = dts_t
        obs = torch.stack([self.obs(f) for f in feats])  # (T,B,C,H,W)
        _obs_norm: nn.LayerNorm | None = self.obs_norm
        _vf_norm: nn.LayerNorm | None = self.vf_norm
        if self.balance and _obs_norm is not None:
            # per-location channel LayerNorm: energy alignment
            t_, b_, c_, h_, w_ = obs.shape
            tb = obs.permute(0, 1, 3, 4, 2).reshape(t_ * b_ * h_ * w_, c_)
            obs = _obs_norm(tb).reshape(t_, b_, h_, w_, c_).permute(0, 1, 4, 2, 3)
        states, preds, gates = [obs[0]], [], []
        for i in range(1, t):
            vf_out = self.vf(states[-1])
            if self.balance and _vf_norm is not None:
                b_, c_, h_, w_ = vf_out.shape
                tb = vf_out.permute(0, 2, 3, 1).reshape(b_ * h_ * w_, c_)
                vf_out = _vf_norm(tb).reshape(b_, h_, w_, c_).permute(0, 3, 1, 2)
            assert isinstance(dts, torch.Tensor)
            s_bar = states[-1] + dts[i - 1] * vf_out  # Eq.7 Euler
            k = self.gate(s_bar, obs[i])  # Eq.8
            s = (1 - k) * s_bar + k * obs[i]  # Eq.9
            preds.append(s_bar)
            gates.append(k)
            states.append(s)
        z = torch.stack(states)
        empty = obs[:0]
        return {
            "states": z,
            "preds": torch.stack(preds) if preds else empty,
            "obs": obs,
            "gates": torch.stack(gates) if gates else empty,
        }


class PCLF(nn.Module):
    """Dual-scale PCLF: forward(F_tf[T], F_tc[T], dts) -> z_t=(S_tc, S_tf)."""

    def __init__(self, channels_fine: int, channels_coarse: int | None = None,
                 vf_deep: bool = False, k_temp: float = 1.0,
                 balance: bool = False) -> None:
        super().__init__()
        self.flow_f = ScaleFlow(channels_fine, vf_deep=vf_deep, k_temp=k_temp,
                                balance=balance)
        self.flow_c = ScaleFlow(channels_coarse or channels_fine, vf_deep=vf_deep,
                                k_temp=k_temp, balance=balance)

    def forward(
        self,
        f_tf: torch.Tensor,
        f_tc: torch.Tensor,
        dts: torch.Tensor | float = 1.0,
    ) -> dict[str, dict[str, torch.Tensor]]:
        """f_tf/f_tc: (T, B, C, H, W) target-frame feature sequences."""
        return {"fine": self.flow_f(f_tf, dts), "coarse": self.flow_c(f_tc, dts)}

    def states(
        self, f_tf: torch.Tensor, f_tc: torch.Tensor, dts: torch.Tensor | float = 1.0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Shortcut -> (S_tf, S_tc), each (T,B,C,H,W), for HDC readouts."""
        out = self.forward(f_tf, f_tc, dts)
        return out["fine"]["states"], out["coarse"]["states"]
