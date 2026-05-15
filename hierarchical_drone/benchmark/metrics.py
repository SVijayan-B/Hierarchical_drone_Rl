from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class CompareConfig:
    rounds: int = 8
    round_timeout_sec: float = 40.0
    hit_radius_m: float = 0.60
    stabilize_dist_m: float = 0.30
    stabilize_z_tol_m: float = 0.08
    target_height: float = 1.0
    target_seed: int = 11
    gui: bool = False


@dataclass
class RoundMetrics:
    round: int
    hit_time_sec: Optional[float]
    stabilization_time_sec: Optional[float]
    power_proxy: Optional[float]

    def to_dict(self) -> Dict[str, Optional[float]]:
        return {
            "round": float(self.round),
            "hit_time_sec": self.hit_time_sec,
            "stabilization_time_sec": self.stabilization_time_sec,
            "power_proxy": self.power_proxy,
        }


def generate_targets(cfg: CompareConfig) -> List[List[float]]:
    rng = random.Random(cfg.target_seed)
    return [
        [rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0), cfg.target_height]
        for _ in range(cfg.rounds)
    ]


def finalize_round_metrics(
    round_idx: int,
    hit_time: Optional[float],
    settle_time: Optional[float],
    power_samples: List[float],
    timeout_sec: float,
) -> RoundMetrics:
    power_proxy = float(sum(power_samples) / max(1, len(power_samples))) if power_samples else None
    return RoundMetrics(
        round=round_idx + 1,
        hit_time_sec=hit_time,
        stabilization_time_sec=settle_time,
        power_proxy=power_proxy,
    )
