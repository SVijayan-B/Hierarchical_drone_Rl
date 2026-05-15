"""Orchestrate PID vs RL+PID benchmark and save plots."""

from __future__ import annotations

import os
from datetime import datetime
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np

from hierarchical_drone.benchmark.metrics import CompareConfig, RoundMetrics, generate_targets
from hierarchical_drone.benchmark.pid_baseline import run_pid_rounds
from hierarchical_drone.benchmark.rl_pid_runner import run_rl_pid_rounds


def _values(metrics: List[RoundMetrics], key: str) -> np.ndarray:
    out = []
    for m in metrics:
        val = getattr(m, key)
        out.append(np.nan if val is None else float(val))
    return np.array(out, dtype=float)


def _save_plots(
    output_dir: str,
    rounds_x: np.ndarray,
    pid_metrics: List[RoundMetrics],
    rl_metrics: List[RoundMetrics],
) -> None:
    pid_hit = _values(pid_metrics, "hit_time_sec")
    rl_hit = _values(rl_metrics, "hit_time_sec")
    pid_stab = _values(pid_metrics, "stabilization_time_sec")
    rl_stab = _values(rl_metrics, "stabilization_time_sec")
    pid_power = _values(pid_metrics, "power_proxy")
    rl_power = _values(rl_metrics, "power_proxy")

    plt.figure(figsize=(10, 5))
    plt.plot(rounds_x, pid_hit, marker="o", color="green", label="PID")
    plt.plot(rounds_x, rl_hit, marker="o", color="blue", label="RL+PID")
    plt.title("Time to Reach Target: PID vs RL+PID")
    plt.xlabel("Round")
    plt.ylabel("Time (s)")
    plt.grid(True, alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "hit_time_comparison.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(rounds_x, pid_stab, marker="o", color="green", label="PID")
    plt.plot(rounds_x, rl_stab, marker="o", color="blue", label="RL+PID")
    plt.title("Stability Time: PID vs RL+PID")
    plt.xlabel("Round")
    plt.ylabel("Time (s)")
    plt.grid(True, alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "stabilization_comparison.png"), dpi=150)
    plt.close()

    width = 0.38
    plt.figure(figsize=(10, 5))
    plt.bar(rounds_x - width / 2, pid_power, width=width, color="green", alpha=0.85, label="PID")
    plt.bar(rounds_x + width / 2, rl_power, width=width, color="blue", alpha=0.85, label="RL+PID")
    plt.title("Power Consumption Proxy: PID vs RL+PID")
    plt.xlabel("Round")
    plt.ylabel("Power proxy (normalized RPM²)")
    plt.grid(True, axis="y", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "power_proxy_comparison.png"), dpi=150)
    plt.close()


def _write_csv(
    output_dir: str,
    pid_metrics: List[RoundMetrics],
    rl_metrics: List[RoundMetrics],
) -> str:
    csv_path = os.path.join(output_dir, "metrics_table.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(
            "round,pid_hit_time_sec,rl_pid_hit_time_sec,pid_stabilization_sec,rl_pid_stabilization_sec,"
            "pid_power_proxy,rl_pid_power_proxy\n"
        )
        for pm, rm in zip(pid_metrics, rl_metrics):
            def _cell(v: Optional[float]) -> str:
                return "" if v is None else f"{v}"

            f.write(
                f"{int(pm.round)},"
                f"{_cell(pm.hit_time_sec)},{_cell(rm.hit_time_sec)},"
                f"{_cell(pm.stabilization_time_sec)},{_cell(rm.stabilization_time_sec)},"
                f"{_cell(pm.power_proxy)},{_cell(rm.power_proxy)}\n"
            )
    return csv_path


def _print_summary(pid_metrics: List[RoundMetrics], rl_metrics: List[RoundMetrics]) -> None:
    print("\n=== PID vs RL+PID METRICS ===")
    print(
        "Round | PID_hit(s) | RL+PID_hit(s) | PID_stab(s) | RL+PID_stab(s) | "
        "PID_power | RL+PID_power"
    )
    for pm, rm in zip(pid_metrics, rl_metrics):
        def _fmt(v: Optional[float]) -> str:
            return "timeout" if v is None else f"{v:.2f}"

        print(
            f"{int(pm.round):>5} | {_fmt(pm.hit_time_sec):>10} | {_fmt(rm.hit_time_sec):>13} | "
            f"{_fmt(pm.stabilization_time_sec):>11} | {_fmt(rm.stabilization_time_sec):>14} | "
            f"{_fmt(pm.power_proxy):>9} | {_fmt(rm.power_proxy):>12}"
        )


def run_compare(
    model_path: str,
    vecnorm_path: str,
    cfg: CompareConfig,
    output_root: Optional[str] = None,
) -> str:
    targets = generate_targets(cfg)
    print(f"[INFO] Shared targets ({cfg.rounds} rounds): {targets}")

    print("[INFO] Running PID-only baseline...")
    pid_metrics = run_pid_rounds(targets, cfg)

    print("[INFO] Running RL+PID (PPO + DSLPIDControl)...")
    rl_metrics = run_rl_pid_rounds(targets, cfg, model_path, vecnorm_path)

    _print_summary(pid_metrics, rl_metrics)

    root = output_root or os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "results_pid_vs_rl_pid",
    )
    output_dir = os.path.join(root, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(output_dir, exist_ok=True)

    rounds_x = np.arange(1, cfg.rounds + 1)
    _save_plots(output_dir, rounds_x, pid_metrics, rl_metrics)
    csv_path = _write_csv(output_dir, pid_metrics, rl_metrics)

    print(f"\n[INFO] Saved graphs and {csv_path}")
    print(f"[INFO] Output folder: {output_dir}")
    return output_dir
