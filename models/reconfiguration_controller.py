"""MCR-UAV Reconfiguration Controller.

This module implements the ReconfigurationController interface to manage
multi-tier control parameter updates, validation, and multi-rate PID gain smoothing.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Union, Optional
import numpy as np
import torch

from models.meta_supervisor import ReconfigurationBounds, ReconfigurationVector


class ReconfigurationController:
    """Manages multi-rate parameter reconfiguration and transitions."""

    def __init__(
        self,
        bounds: Optional[ReconfigurationBounds] = None,
        beta_gain: float = 0.05,
    ) -> None:
        self.bounds = bounds if bounds is not None else ReconfigurationBounds()
        self.beta_gain = beta_gain

        # Fallback nominal values
        self.nominal_lambda_rl = 1.0
        self.nominal_alpha_q = 1.0
        self.nominal_alpha_r = 1.0
        self.nominal_alpha_p = 1.0
        self.nominal_alpha_i = 1.0
        self.nominal_alpha_d = 1.0
        self.nominal_horizon = 20

        # Current targets
        self.target_lambda_rl = self.nominal_lambda_rl
        self.target_alpha_q = self.nominal_alpha_q
        self.target_alpha_r = self.nominal_alpha_r
        self.target_alpha_p = self.nominal_alpha_p
        self.target_alpha_i = self.nominal_alpha_i
        self.target_alpha_d = self.nominal_alpha_d
        self.target_horizon = self.nominal_horizon

        # Current effective (smoothed) values
        self.effective_lambda_rl = self.nominal_lambda_rl
        self.effective_alpha_q = self.nominal_alpha_q
        self.effective_alpha_r = self.nominal_alpha_r
        self.effective_alpha_p = self.nominal_alpha_p
        self.effective_alpha_i = self.nominal_alpha_i
        self.effective_alpha_d = self.nominal_alpha_d
        self.effective_horizon = self.nominal_horizon

        self.last_c_t: Optional[Any] = None

    def reset(self) -> None:
        """Reset targets and effective values back to nominal fallback values."""
        self.target_lambda_rl = self.nominal_lambda_rl
        self.target_alpha_q = self.nominal_alpha_q
        self.target_alpha_r = self.nominal_alpha_r
        self.target_alpha_p = self.nominal_alpha_p
        self.target_alpha_i = self.nominal_alpha_i
        self.target_alpha_d = self.nominal_alpha_d
        self.target_horizon = self.nominal_horizon

        # Reset effective values immediately to avoid historical carryover
        self.effective_lambda_rl = self.nominal_lambda_rl
        self.effective_alpha_q = self.nominal_alpha_q
        self.effective_alpha_r = self.nominal_alpha_r
        self.effective_alpha_p = self.nominal_alpha_p
        self.effective_alpha_i = self.nominal_alpha_i
        self.effective_alpha_d = self.nominal_alpha_d
        self.effective_horizon = self.nominal_horizon

        self.last_c_t = None

    def apply_reconfiguration(self, c_t: Any) -> bool:
        """Receive, validate, and latch a new reconfiguration command.

        Returns True if the configuration was valid, False if fallback was triggered.
        """
        if c_t is None:
            self._apply_fallback()
            return False

        try:
            parsed = self._parse_c_t(c_t)
            if parsed is None or not self._validate_config(parsed):
                self._apply_fallback()
                return False

            self.last_c_t = c_t
            self.target_lambda_rl = parsed["lambda_rl"]
            self.target_alpha_q = parsed["alpha_q"]
            self.target_alpha_r = parsed["alpha_r"]
            self.target_alpha_p = parsed["alpha_p"]
            self.target_alpha_i = parsed["alpha_i"]
            self.target_alpha_d = parsed["alpha_d"]
            self.target_horizon = int(parsed["horizon"])

            # Non-PID variables switch instantaneously at the 10 Hz rate
            self.effective_lambda_rl = self.target_lambda_rl
            self.effective_alpha_q = self.target_alpha_q
            self.effective_alpha_r = self.target_alpha_r
            self.effective_horizon = self.target_horizon
            return True

        except Exception:
            self._apply_fallback()
            return False

    def smooth_inner_step(self) -> None:
        """Update effective smoothed PID gains at 1200 Hz."""
        # K_new = (1 - beta) * K_old + beta * K_target
        self.effective_alpha_p = (1.0 - self.beta_gain) * self.effective_alpha_p + self.beta_gain * self.target_alpha_p
        self.effective_alpha_i = (1.0 - self.beta_gain) * self.effective_alpha_i + self.beta_gain * self.target_alpha_i
        self.effective_alpha_d = (1.0 - self.beta_gain) * self.effective_alpha_d + self.beta_gain * self.target_alpha_d

        # Clip as final safety guard against numerical drift
        self.effective_alpha_p = float(np.clip(self.effective_alpha_p, self.bounds.alpha_p_min, self.bounds.alpha_p_max))
        self.effective_alpha_i = float(np.clip(self.effective_alpha_i, self.bounds.alpha_i_min, self.bounds.alpha_i_max))
        self.effective_alpha_d = float(np.clip(self.effective_alpha_d, self.bounds.alpha_d_min, self.bounds.alpha_d_max))

    def get_effective_params(self) -> Dict[str, Any]:
        """Return the currently effective (smoothed) configuration parameters."""
        return {
            "lambda_rl": self.effective_lambda_rl,
            "alpha_q": self.effective_alpha_q,
            "alpha_r": self.effective_alpha_r,
            "alpha_p": self.effective_alpha_p,
            "alpha_i": self.effective_alpha_i,
            "alpha_d": self.effective_alpha_d,
            "horizon": self.effective_horizon,
        }

    def _apply_fallback(self) -> None:
        """Revert targets and instantaneous effective parameters to nominal fallback values."""
        self.target_lambda_rl = self.nominal_lambda_rl
        self.target_alpha_q = self.nominal_alpha_q
        self.target_alpha_r = self.nominal_alpha_r
        self.target_alpha_p = self.nominal_alpha_p
        self.target_alpha_i = self.nominal_alpha_i
        self.target_alpha_d = self.nominal_alpha_d
        self.target_horizon = self.nominal_horizon

        # Non-PID params revert instantly
        self.effective_lambda_rl = self.nominal_lambda_rl
        self.effective_alpha_q = self.nominal_alpha_q
        self.effective_alpha_r = self.nominal_alpha_r
        self.effective_horizon = self.nominal_horizon

    def _parse_c_t(self, c_t: Any) -> Optional[Dict[str, float]]:
        """Unified parser to convert ReconfigurationVector, dict, list, or array to a floats dict."""
        if isinstance(c_t, ReconfigurationVector):
            data = c_t.to_numpy()
            return {
                "lambda_rl": float(np.ravel(data["lambda_rl"])[0]),
                "alpha_q": float(np.ravel(data["alpha_q"])[0]),
                "alpha_r": float(np.ravel(data["alpha_r"])[0]),
                "alpha_p": float(np.ravel(data["alpha_p"])[0]),
                "alpha_i": float(np.ravel(data["alpha_i"])[0]),
                "alpha_d": float(np.ravel(data["alpha_d"])[0]),
                "horizon": float(np.ravel(data["horizon"])[0]),
            }

        if isinstance(c_t, dict):
            # Resolve tensors or numpy types inside dict
            parsed = {}
            for k in ["lambda_rl", "alpha_q", "alpha_r", "alpha_p", "alpha_i", "alpha_d", "horizon"]:
                if k not in c_t:
                    return None
                val = c_t[k]
                if isinstance(val, (torch.Tensor, np.ndarray)):
                    parsed[k] = float(np.ravel(val)[0])
                else:
                    parsed[k] = float(val)
            return parsed

        # Sequence types (list, tuple, np.ndarray, torch.Tensor)
        if isinstance(c_t, (list, tuple, np.ndarray, torch.Tensor)):
            if isinstance(c_t, torch.Tensor):
                arr = c_t.detach().cpu().numpy().ravel()
            else:
                arr = np.ravel(c_t)

            if len(arr) < 7:
                return None

            return {
                "lambda_rl": float(arr[0]),
                "alpha_q": float(arr[1]),
                "alpha_r": float(arr[2]),
                "alpha_p": float(arr[3]),
                "alpha_i": float(arr[4]),
                "alpha_d": float(arr[5]),
                "horizon": float(arr[6]),
            }

        return None

    def _validate_config(self, parsed: Dict[str, float]) -> bool:
        """Validate parameter ranges and finite-state assertions."""
        # 1. NaN and Inf check
        for val in parsed.values():
            if not math.isfinite(val):
                return False

        # 2. Check bounds
        if not (self.bounds.lambda_rl_min - 1e-7 <= parsed["lambda_rl"] <= self.bounds.lambda_rl_max + 1e-7):
            return False
        if not (self.bounds.alpha_q_min - 1e-7 <= parsed["alpha_q"] <= self.bounds.alpha_q_max + 1e-7):
            return False
        if not (self.bounds.alpha_r_min - 1e-7 <= parsed["alpha_r"] <= self.bounds.alpha_r_max + 1e-7):
            return False
        if not (self.bounds.alpha_p_min - 1e-7 <= parsed["alpha_p"] <= self.bounds.alpha_p_max + 1e-7):
            return False
        if not (self.bounds.alpha_i_min - 1e-7 <= parsed["alpha_i"] <= self.bounds.alpha_i_max + 1e-7):
            return False
        if not (self.bounds.alpha_d_min - 1e-7 <= parsed["alpha_d"] <= self.bounds.alpha_d_max + 1e-7):
            return False

        # 3. Horizon must be strictly in options
        horizon_val = int(round(parsed["horizon"]))
        if horizon_val not in self.bounds.horizon_options:
            return False

        # If it is fractional, e.g. 19.8, but close enough to 20, round it. But if it's 17 or 24, reject.
        if abs(parsed["horizon"] - horizon_val) > 1e-4:
            return False

        return True
