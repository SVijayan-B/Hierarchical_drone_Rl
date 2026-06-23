from dataclasses import dataclass


@dataclass
class CompareConfig:
    rounds: int
    round_timeout_sec: float
    target_seed: int
    gui: bool
    hit_radius_m: float = 0.60
    domain_randomization: bool = False
