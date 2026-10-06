"""MCR-UAV Standalone Meta-Supervisor Architecture.

This module implements the Meta-Reconfiguration Supervisor ($g_\\theta$)
for the Meta-Contextual Reconfiguration (MCR-UAV) framework.

Mathematical Formulation:
- Input latent context: z_t \\in \\mathbb{R}^{B \\times 16}
- Shared feature extraction:
    h_1 = GELU(LayerNorm(W_1 z_t + b_1)),  W_1 \\in \\mathbb{R}^{64 \\times 16}
    h_2 = GELU(W_2 h_1 + b_2),             W_2 \\in \\mathbb{R}^{64 \\times 64}
- Continuous Multi-Tier Parameter Heads:
    y_i = y_{i,\\min} + \\sigma(W_i h_2 + b_i) \\cdot (y_{i,\\max} - y_{i,\\min})
    c_t^{\\text{cont}} = [\\lambda_{\\text{RL}}, \\alpha_Q, \\alpha_R, \\alpha_P, \\alpha_I, \\alpha_D]
- Discrete Categorical MPC Horizon Head:
    \\pi_H = \\text{softmax}(W_H h_2 + b_H),   W_H \\in \\mathbb{R}^{3 \\times 64}
    H_t \\in \\{10, 20, 30\\}
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ReconfigurationBounds:
    """Configurable physical bounds for multi-tier parameter scaling."""

    lambda_rl_min: float = 0.0
    lambda_rl_max: float = 1.0

    alpha_q_min: float = 0.2
    alpha_q_max: float = 5.0

    alpha_r_min: float = 0.2
    alpha_r_max: float = 5.0

    alpha_p_min: float = 0.5
    alpha_p_max: float = 2.0

    alpha_i_min: float = 0.2
    alpha_i_max: float = 2.5

    alpha_d_min: float = 0.5
    alpha_d_max: float = 2.0

    horizon_options: Tuple[int, ...] = (10, 20, 30)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lambda_rl": (self.lambda_rl_min, self.lambda_rl_max),
            "alpha_q": (self.alpha_q_min, self.alpha_q_max),
            "alpha_r": (self.alpha_r_min, self.alpha_r_max),
            "alpha_p": (self.alpha_p_min, self.alpha_p_max),
            "alpha_i": (self.alpha_i_min, self.alpha_i_max),
            "alpha_d": (self.alpha_d_min, self.alpha_d_max),
            "horizon_options": list(self.horizon_options),
        }


@dataclass
class ReconfigurationVector:
    """Structured container for the multi-tier reconfiguration parameters."""

    lambda_rl: torch.Tensor
    alpha_q: torch.Tensor
    alpha_r: torch.Tensor
    alpha_p: torch.Tensor
    alpha_i: torch.Tensor
    alpha_d: torch.Tensor
    horizon_logits: torch.Tensor
    horizon_probabilities: torch.Tensor
    horizon: torch.Tensor
    continuous_vector: torch.Tensor

    def as_dict(self) -> Dict[str, torch.Tensor]:
        return {
            "lambda_rl": self.lambda_rl,
            "alpha_q": self.alpha_q,
            "alpha_r": self.alpha_r,
            "alpha_p": self.alpha_p,
            "alpha_i": self.alpha_i,
            "alpha_d": self.alpha_d,
            "horizon_logits": self.horizon_logits,
            "horizon_probabilities": self.horizon_probabilities,
            "horizon": self.horizon,
            "continuous_vector": self.continuous_vector,
        }

    def to_numpy(self) -> Dict[str, Any]:
        return {
            "lambda_rl": self.lambda_rl.detach().cpu().numpy(),
            "alpha_q": self.alpha_q.detach().cpu().numpy(),
            "alpha_r": self.alpha_r.detach().cpu().numpy(),
            "alpha_p": self.alpha_p.detach().cpu().numpy(),
            "alpha_i": self.alpha_i.detach().cpu().numpy(),
            "alpha_d": self.alpha_d.detach().cpu().numpy(),
            "horizon_logits": self.horizon_logits.detach().cpu().numpy(),
            "horizon_probabilities": self.horizon_probabilities.detach().cpu().numpy(),
            "horizon": self.horizon.detach().cpu().numpy(),
            "continuous_vector": self.continuous_vector.detach().cpu().numpy(),
        }


class MetaSupervisor(nn.Module):
    """Standalone Meta-Supervisor mapping latent context z_t to reconfiguration parameters."""

    def __init__(
        self,
        latent_dim: int = 16,
        hidden_dim: int = 64,
        bounds: Optional[ReconfigurationBounds] = None,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.bounds = bounds if bounds is not None else ReconfigurationBounds()

        # Shared Feature Extraction Trunk
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.act1 = nn.GELU()

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.act2 = nn.GELU()

        # 6 Continuous Output Heads
        self.head_lambda_rl = nn.Linear(hidden_dim, 1)
        self.head_alpha_q = nn.Linear(hidden_dim, 1)
        self.head_alpha_r = nn.Linear(hidden_dim, 1)
        self.head_alpha_p = nn.Linear(hidden_dim, 1)
        self.head_alpha_i = nn.Linear(hidden_dim, 1)
        self.head_alpha_d = nn.Linear(hidden_dim, 1)

        # 1 Discrete Categorical Head for MPC Prediction Horizon H
        self.num_horizons = len(self.bounds.horizon_options)
        self.head_horizon = nn.Linear(hidden_dim, self.num_horizons)

        # Register horizon options as buffer
        horizon_tensor = torch.tensor(self.bounds.horizon_options, dtype=torch.long)
        self.register_buffer("horizon_options_tensor", horizon_tensor)

    def _validate_input(self, z: torch.Tensor) -> None:
        """Validate input tensor sanity (finite values, dimensional consistency)."""
        if not torch.is_floating_point(z):
            raise TypeError(f"Latent context tensor must be floating point, got {z.dtype}")

        if torch.isnan(z).any():
            raise ValueError("Input latent context z contains NaN values; rejecting numerical corruption")

        if torch.isinf(z).any():
            raise ValueError("Input latent context z contains Inf values; rejecting numerical explosion")

        if z.dim() == 1:
            if z.size(0) != self.latent_dim:
                raise ValueError(
                    f"Expected latent dimension {self.latent_dim}, got {z.size(0)}"
                )
        elif z.dim() == 2:
            if z.size(1) != self.latent_dim:
                raise ValueError(
                    f"Expected latent dimension {self.latent_dim}, got {z.size(1)}"
                )
        else:
            raise ValueError(
                f"Expected 1D [16] or 2D [B, 16] latent context tensor, got shape {z.shape}"
            )

    def _scale_bounded(
        self,
        raw_output: torch.Tensor,
        val_min: float,
        val_max: float,
    ) -> torch.Tensor:
        """Scale unbounded neural output to strict physical interval [val_min, val_max]."""
        return val_min + torch.sigmoid(raw_output) * (val_max - val_min)

    def forward(
        self,
        z: torch.Tensor,
        return_hard: bool = True,
    ) -> ReconfigurationVector:
        """Forward pass computing multi-tier reconfiguration parameters.
        
        Parameters
        ----------
        z : torch.Tensor
            Latent context tensor of shape [B, 16] or [16].
        return_hard : bool
            If True, extracts the discrete horizon value via argmax.
            
        Returns
        -------
        ReconfigurationVector
            Structured object containing bounded continuous parameters and discrete horizon.
        """
        self._validate_input(z)

        # Handle 1D unbatched input [16] -> [1, 16]
        is_unbatched = z.dim() == 1
        if is_unbatched:
            z = z.unsqueeze(0)

        # Shared Trunk Forward Pass
        h1 = self.act1(self.ln1(self.fc1(z)))
        h2 = self.act2(self.fc2(h1))

        # 1. High-Level RL Blending Authority: lambda_RL in [0.0, 1.0]
        raw_lambda = self.head_lambda_rl(h2)
        lambda_rl = self._scale_bounded(
            raw_lambda, self.bounds.lambda_rl_min, self.bounds.lambda_rl_max
        )

        # 2. Mid-Level MPC Optimization Weight Scaling: alpha_Q, alpha_R in [0.2, 5.0]
        raw_q = self.head_alpha_q(h2)
        alpha_q = self._scale_bounded(
            raw_q, self.bounds.alpha_q_min, self.bounds.alpha_q_max
        )

        raw_r = self.head_alpha_r(h2)
        alpha_r = self._scale_bounded(
            raw_r, self.bounds.alpha_r_min, self.bounds.alpha_r_max
        )

        # 3. Low-Level PID Tracking Gain Scaling: alpha_P in [0.5, 2.0], alpha_I in [0.2, 2.5], alpha_D in [0.5, 2.0]
        raw_p = self.head_alpha_p(h2)
        alpha_p = self._scale_bounded(
            raw_p, self.bounds.alpha_p_min, self.bounds.alpha_p_max
        )

        raw_i = self.head_alpha_i(h2)
        alpha_i = self._scale_bounded(
            raw_i, self.bounds.alpha_i_min, self.bounds.alpha_i_max
        )

        raw_d = self.head_alpha_d(h2)
        alpha_d = self._scale_bounded(
            raw_d, self.bounds.alpha_d_min, self.bounds.alpha_d_max
        )

        # 4. Mid-Level Discrete MPC Prediction Horizon Head: H in {10, 20, 30}
        horizon_logits = self.head_horizon(h2)  # [B, 3]
        horizon_probs = F.softmax(horizon_logits, dim=-1)

        if return_hard:
            horizon_indices = torch.argmax(horizon_logits, dim=-1)
            selected_horizon = self.horizon_options_tensor[horizon_indices].unsqueeze(-1)
        else:
            # Soft expected horizon (differentiable training proxy)
            selected_horizon = torch.sum(
                horizon_probs * self.horizon_options_tensor.float().unsqueeze(0),
                dim=-1,
                keepdim=True,
            )

        # Concatenate 6 continuous parameters: [B, 6]
        continuous_vector = torch.cat(
            [lambda_rl, alpha_q, alpha_r, alpha_p, alpha_i, alpha_d], dim=-1
        )

        return ReconfigurationVector(
            lambda_rl=lambda_rl,
            alpha_q=alpha_q,
            alpha_r=alpha_r,
            alpha_p=alpha_p,
            alpha_i=alpha_i,
            alpha_d=alpha_d,
            horizon_logits=horizon_logits,
            horizon_probabilities=horizon_probs,
            horizon=selected_horizon,
            continuous_vector=continuous_vector,
        )

    def count_parameters(self) -> int:
        """Return total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def parameter_breakdown(self) -> Dict[str, int]:
        """Return a structured breakdown of trainable parameter counts per layer/head."""
        return {
            "input_network": (
                self.fc1.weight.numel()
                + self.fc1.bias.numel()
                + self.ln1.weight.numel()
                + self.ln1.bias.numel()
                + self.fc2.weight.numel()
                + self.fc2.bias.numel()
            ),
            "lambda_head": self.head_lambda_rl.weight.numel() + self.head_lambda_rl.bias.numel(),
            "alpha_q_head": self.head_alpha_q.weight.numel() + self.head_alpha_q.bias.numel(),
            "alpha_r_head": self.head_alpha_r.weight.numel() + self.head_alpha_r.bias.numel(),
            "alpha_p_head": self.head_alpha_p.weight.numel() + self.head_alpha_p.bias.numel(),
            "alpha_i_head": self.head_alpha_i.weight.numel() + self.head_alpha_i.bias.numel(),
            "alpha_d_head": self.head_alpha_d.weight.numel() + self.head_alpha_d.bias.numel(),
            "horizon_head": self.head_horizon.weight.numel() + self.head_horizon.bias.numel(),
            "total": self.count_parameters(),
        }
