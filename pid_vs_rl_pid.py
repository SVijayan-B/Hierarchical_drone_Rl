"""Entry point: compare PID-only vs RL+PID and save graphs.

Run with conda env:
  conda activate drones
  python pid_vs_rl_pid.py --rounds 8
"""

from __future__ import annotations

import argparse
import os

from hierarchical_drone.benchmark.compare_runner import run_compare
from hierarchical_drone.benchmark.metrics import CompareConfig


def main() -> None:
    default_run = os.path.join("results_hierarchical", "run_20260513_112909")
    parser = argparse.ArgumentParser(description="Compare PID-only vs RL+PID hierarchical stack")
    parser.add_argument("--model", type=str, default=os.path.join(default_run, "final_model.zip"))
    parser.add_argument("--vecnorm", type=str, default=os.path.join(default_run, "vecnormalize.pkl"))
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--timeout", type=float, default=40.0)
    args = parser.parse_args()

    if not os.path.isfile(args.model):
        raise FileNotFoundError(f"Model not found: {args.model}")
    if not os.path.isfile(args.vecnorm):
        raise FileNotFoundError(f"VecNormalize not found: {args.vecnorm}")

    cfg = CompareConfig(
        rounds=args.rounds,
        round_timeout_sec=args.timeout,
        target_seed=args.seed,
        gui=args.gui,
        hit_radius_m=0.60,
    )
    run_compare(args.model, args.vecnorm, cfg)


if __name__ == "__main__":
    main()
