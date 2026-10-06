"""MCR-UAV Phase 5: Meta-Train / Validation / OOD Test Evaluation Protocol.

This module implements the formal evaluation protocol for benchmarking the frozen
Honors baseline models (MLP PPO vs. Transformer PPO) across the 130-task distribution.

Key Features:
1. Manifest integrity & leakage assertions (80 Train, 20 Val, 30 OOD).
2. Clean baseline model loading with exact VecNormalize statistics.
3. Paired evaluation across identical task instances and seeds.
4. Comprehensive metric recording: RMSE, overshoot, peak/avg jerk, energy, control effort.
5. Export of per-episode results to results/phase5_baseline_evaluation.csv.
6. Export of aggregated statistical summary to results/phase5_baseline_summary.csv.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

try:
    import yaml
except ImportError:
    yaml = None

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from environments.task_generator import MetaTask, TaskGenerator, TaskSplit
from evaluation.validate_tasks import TaskValidator
from hierarchical_drone.config.settings import (
    ActionConfig,
    DroneConfig,
    SensorConfig,
    SimConfig,
    TaskConfig,
)
from hierarchical_drone.env.hierarchical_nav_env import HierarchicalNavEnv


class EvaluationProtocol:
    """Standardized evaluation protocol for baseline and Meta-RL UAV models."""

    EVAL_CSV_HEADER = [
        "model",
        "split",
        "task_id",
        "seed",
        "episode",
        "success",
        "reward",
        "rmse_tracking_error",
        "mean_tracking_error",
        "max_position_error",
        "settling_time",
        "overshoot",
        "overshoot_percent",
        "peak_jerk",
        "avg_jerk",
        "control_effort",
        "energy",
        "distance_traveled",
        "collision",
        "fell_down",
        "episode_length",
    ]

    SUMMARY_CSV_HEADER = [
        "model",
        "split",
        "task_count",
        "mean_success_rate",
        "std_success_rate",
        "mean_reward",
        "std_reward",
        "mean_rmse",
        "std_rmse",
        "mean_settling_time",
        "std_settling_time",
        "mean_overshoot",
        "std_overshoot",
        "mean_peak_jerk",
        "std_peak_jerk",
        "mean_energy",
        "std_energy",
        "mean_control_effort",
        "std_control_effort",
    ]

    def __init__(
        self,
        config_path: str = "configs/evaluation_protocol.yaml",
        manifest_path: str = "configs/meta_tasks_manifest.json",
        gui: bool = False,
    ) -> None:
        """Initialize the EvaluationProtocol."""
        self.config_path = config_path
        self.manifest_path = manifest_path
        self.gui = gui
        
        # Load protocol configuration (with JSON companion fallback)
        self.cfg = self._load_config(self.config_path)

        # Load and validate manifest
        self.manifest = self._load_and_validate_manifest()

    def _load_config(self, path: str) -> Dict[str, Any]:
        """Load YAML/JSON configuration."""
        if path.endswith(".yaml") or path.endswith(".yml"):
            if yaml is not None:
                with open(path, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f)
            # Try loading corresponding .json file
            json_alt = path.rsplit(".", 1)[0] + ".json"
            if os.path.exists(json_alt):
                with open(json_alt, "r", encoding="utf-8") as f:
                    return json.load(f)
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_and_validate_manifest(self) -> Dict[str, List[MetaTask]]:
        """Load and strictly validate the 130-task manifest."""
        if not os.path.exists(self.manifest_path):
            raise FileNotFoundError(f"Task manifest not found: {self.manifest_path}")

        gen = TaskGenerator(master_seed=self.cfg["reproducibility"]["global_seed"])
        manifest = gen.load_manifest(self.manifest_path)
        gen.validate_task_manifest(manifest)
        return manifest

    def _make_env_fn(self, task: MetaTask, use_history: bool, episode_sec: float = 6.0):
        """Create a callable factory for initializing HierarchicalNavEnv with task disturbance."""
        def _init() -> HierarchicalNavEnv:
            sim_cfg = SimConfig(
                gui=self.gui,
                episode_sec=episode_sec,
                pyb_freq=240,
                ctrl_freq=120,
                rl_freq=10,
            )
            task_cfg = TaskConfig(
                target_threshold_m=0.60,
                world_xy_limit=2.5,
                world_z_min=0.1,
                world_z_max=2.2,
            )
            sensor_cfg = SensorConfig()
            action_cfg = ActionConfig()

            env = HierarchicalNavEnv(
                sim=sim_cfg,
                task=task_cfg,
                sensor_cfg=sensor_cfg,
                action_cfg=action_cfg,
                use_adaptive_scheduler=True,
                demo_guided_mode=False,
                use_mpc_layer=True,
                use_rl_gain_scheduler=False,
                use_adaptive_mpc=False,
                use_history=use_history,
                domain_randomization=False,
            )
            env.evaluation_mode = True
            return env
        return _init

    def evaluate_episode(
        self,
        model: PPO,
        vecnorm_path: str,
        task: MetaTask,
        model_name: str,
        use_history: bool,
        episode_idx: int = 1,
        episode_sec: float = 6.0,
    ) -> Dict[str, Any]:
        """Evaluate a single episode on a specific MetaTask."""
        env_fn = self._make_env_fn(task=task, use_history=use_history, episode_sec=episode_sec)
        vec_env = DummyVecEnv([env_fn])
        
        # Load normalization stats
        vec_env = VecNormalize.load(vecnorm_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False

        raw_env: HierarchicalNavEnv = vec_env.envs[0]

        # Inject task disturbances
        validator = TaskValidator()
        validator.inject_task(raw_env, task)

        obs = vec_env.reset()
        done = False
        ep_reward = 0.0
        step_count = 0

        info_dict: Dict[str, Any] = {}

        try:
            while not done:
                step_count += 1
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, dones, infos = vec_env.step(action)
                done = dones[0]
                ep_reward += float(reward[0])
                info_dict = infos[0]
        finally:
            try:
                vec_env.close()
            except Exception:
                pass

        success_val = 1 if info_dict.get("success", 0.0) > 0.5 else 0
        rmse = float(info_dict.get("rmse_tracking_error", math.nan))
        mean_err = float(info_dict.get("mean_tracking_error", math.nan))
        max_err = float(info_dict.get("max_position_error", math.nan))
        settling_time = float(info_dict.get("settling_time", math.nan))
        overshoot = float(info_dict.get("overshoot", math.nan))
        overshoot_pct = float(info_dict.get("overshoot_percent", math.nan))
        peak_jerk = float(info_dict.get("peak_jerk", math.nan))
        avg_jerk = float(info_dict.get("avg_jerk", math.nan))
        ctrl_effort = float(info_dict.get("control_effort", math.nan))
        energy = float(info_dict.get("total_energy", math.nan))
        dist_traveled = float(info_dict.get("distance_traveled", math.nan))
        collision = int(info_dict.get("collision", 0.0) > 0.5)
        fell_down = int(info_dict.get("fell_down", 0.0) > 0.5)

        record: Dict[str, Any] = {
            "model": model_name,
            "split": task.split,
            "task_id": task.task_id,
            "seed": task.seed,
            "episode": episode_idx,
            "success": success_val,
            "reward": round(ep_reward, 3),
            "rmse_tracking_error": round(rmse, 4) if not math.isnan(rmse) else "",
            "mean_tracking_error": round(mean_err, 4) if not math.isnan(mean_err) else "",
            "max_position_error": round(max_err, 4) if not math.isnan(max_err) else "",
            "settling_time": round(settling_time, 3) if not math.isnan(settling_time) else "",
            "overshoot": round(overshoot, 4) if not math.isnan(overshoot) else "",
            "overshoot_percent": round(overshoot_pct, 2) if not math.isnan(overshoot_pct) else "",
            "peak_jerk": round(peak_jerk, 2) if not math.isnan(peak_jerk) else "",
            "avg_jerk": round(avg_jerk, 2) if not math.isnan(avg_jerk) else "",
            "control_effort": round(ctrl_effort, 6) if not math.isnan(ctrl_effort) else "",
            "energy": round(energy, 2) if not math.isnan(energy) else "",
            "distance_traveled": round(dist_traveled, 3) if not math.isnan(dist_traveled) else "",
            "collision": collision,
            "fell_down": fell_down,
            "episode_length": step_count,
        }
        return record

    def evaluate_model(
        self,
        model_name: str,
        tasks: List[MetaTask],
        episodes_per_task: int = 1,
        episode_sec: float = 6.0,
    ) -> List[Dict[str, Any]]:
        """Evaluate a designated baseline model over a given list of MetaTasks."""
        checkpoint_cfg = self.cfg["baseline_checkpoints"][model_name]
        model_path = checkpoint_cfg["model_path"]
        vecnorm_path = checkpoint_cfg["vecnorm_path"]
        use_history = checkpoint_cfg["use_history"]

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Baseline model not found: {model_path}")
        if not os.path.exists(vecnorm_path):
            raise FileNotFoundError(f"VecNormalize pickle not found: {vecnorm_path}")

        print(f"\n[EvaluationProtocol] Loading {model_name.upper()} baseline:")
        print(f"  Model:       {model_path}")
        print(f"  VecNormalize:{vecnorm_path}")
        print(f"  Use History: {use_history}")

        model = PPO.load(model_path)
        records: List[Dict[str, Any]] = []
        total = len(tasks) * episodes_per_task

        idx = 0
        for task in tasks:
            for ep in range(1, episodes_per_task + 1):
                idx += 1
                rec = self.evaluate_episode(
                    model=model,
                    vecnorm_path=vecnorm_path,
                    task=task,
                    model_name=model_name,
                    use_history=use_history,
                    episode_idx=ep,
                    episode_sec=episode_sec,
                )
                records.append(rec)
                succ_str = "SUCCESS" if rec["success"] == 1 else "FAIL"
                print(f"[{idx:03d}/{total:03d}] {model_name.upper():12s} | {task.split:8s} | {task.task_id:18s} | {succ_str} | Rew: {rec['reward']:6.1f} | RMSE: {rec['rmse_tracking_error']}m")

        return records

    def compute_summary_statistics(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Compute aggregated mean and std summary metrics partitioned by model and split."""
        models = sorted(list(set(r["model"] for r in records)))
        splits = ["train", "val", "ood_test"]
        summary_rows: List[Dict[str, Any]] = []

        for m in models:
            for s in splits:
                group = [r for r in records if r["model"] == m and r["split"] == s]
                if not group:
                    continue

                def _safe_stats(key: str) -> Tuple[float, float]:
                    vals = [float(r[key]) for r in group if r[key] != "" and not math.isnan(float(r[key]))]
                    if not vals:
                        return float("nan"), float("nan")
                    return float(np.mean(vals)), float(np.std(vals))

                mean_succ, std_succ = _safe_stats("success")
                mean_rew, std_rew = _safe_stats("reward")
                mean_rmse, std_rmse = _safe_stats("rmse_tracking_error")
                mean_settle, std_settle = _safe_stats("settling_time")
                mean_over, std_over = _safe_stats("overshoot")
                mean_pjerk, std_pjerk = _safe_stats("peak_jerk")
                mean_energy, std_energy = _safe_stats("energy")
                mean_effort, std_effort = _safe_stats("control_effort")

                summary_rows.append({
                    "model": m,
                    "split": s,
                    "task_count": len(group),
                    "mean_success_rate": round(mean_succ, 4),
                    "std_success_rate": round(std_succ, 4),
                    "mean_reward": round(mean_rew, 3),
                    "std_reward": round(std_rew, 3),
                    "mean_rmse": round(mean_rmse, 4),
                    "std_rmse": round(std_rmse, 4),
                    "mean_settling_time": round(mean_settle, 3),
                    "std_settling_time": round(std_settle, 3),
                    "mean_overshoot": round(mean_over, 4) if not math.isnan(mean_over) else "",
                    "std_overshoot": round(std_over, 4) if not math.isnan(std_over) else "",
                    "mean_peak_jerk": round(mean_pjerk, 2),
                    "std_peak_jerk": round(std_pjerk, 2),
                    "mean_energy": round(mean_energy, 2),
                    "std_energy": round(std_energy, 2),
                    "mean_control_effort": round(mean_effort, 6),
                    "std_control_effort": round(std_effort, 6),
                })
        return summary_rows

    def run_full_protocol(
        self,
        out_eval_csv: str = "results/phase5_baseline_evaluation.csv",
        out_summary_csv: str = "results/phase5_baseline_summary.csv",
        smoke_subset: bool = False,
        episodes_per_task: int = 1,
        episode_sec: float = 6.0,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Execute the full evaluation protocol across all models and splits."""
        if smoke_subset:
            tasks = self.manifest["train"][:3] + self.manifest["val"][:2] + self.manifest["ood_test"][:3]
            print(f"\n============================================================")
            print(f"RUNNING SMOKE EVALUATION: 8 tasks per model (16 episodes total)")
            print(f"============================================================")
        else:
            tasks = self.manifest["train"] + self.manifest["val"] + self.manifest["ood_test"]
            print(f"\n============================================================")
            print(f"RUNNING COMPLETE EVALUATION: 130 tasks per model (260 episodes total)")
            print(f"============================================================")

        all_records: List[Dict[str, Any]] = []

        t0 = time.time()
        for model_name in ["mlp", "transformer"]:
            recs = self.evaluate_model(
                model_name=model_name,
                tasks=tasks,
                episodes_per_task=episodes_per_task,
                episode_sec=episode_sec,
            )
            all_records.extend(recs)
        elapsed = time.time() - t0

        # Save per-episode evaluation results
        os.makedirs(os.path.dirname(out_eval_csv), exist_ok=True)
        with open(out_eval_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.EVAL_CSV_HEADER)
            writer.writeheader()
            for r in all_records:
                writer.writerow(r)
        print(f"\n[EvaluationProtocol] Saved per-episode results to: {out_eval_csv}")

        # Compute and save summary statistics
        summary_rows = self.compute_summary_statistics(all_records)
        os.makedirs(os.path.dirname(out_summary_csv), exist_ok=True)
        with open(out_summary_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.SUMMARY_CSV_HEADER)
            writer.writeheader()
            for r in summary_rows:
                writer.writerow(r)
        print(f"[EvaluationProtocol] Saved summary statistics to: {out_summary_csv}")

        self.print_protocol_report(summary_rows, elapsed)
        return all_records, summary_rows

    def print_protocol_report(self, summary_rows: List[Dict[str, Any]], elapsed_sec: float) -> None:
        """Print the formatted evaluation report table."""
        print("\n" + "=" * 80)
        print("MCR-UAV PHASE 5 BASELINE EVALUATION PROTOCOL REPORT")
        print("=" * 80)
        print(f"{'Model':12s} | {'Split':8s} | {'N':3s} | {'Success Rate':12s} | {'Mean Reward':11s} | {'Mean RMSE':10s} | {'Energy':8s}")
        print("-" * 80)
        for r in summary_rows:
            succ_pct = f"{r['mean_success_rate']*100:.1f}%"
            print(f"{r['model'].upper():12s} | {r['split']:8s} | {r['task_count']:3d} | {succ_pct:>12s} | {r['mean_reward']:11.2f} | {r['mean_rmse']:10.4f}m | {r['mean_energy']:8.1f}")
        print("-" * 80)
        print(f"Total Execution Time: {elapsed_sec:.2f}s")
        print("=" * 80 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="MCR-UAV Phase 5 Baseline Evaluation Protocol")
    parser.add_argument("--config", type=str, default="configs/evaluation_protocol.yaml", help="Path to protocol YAML")
    parser.add_argument("--manifest", type=str, default="configs/meta_tasks_manifest.json", help="Path to tasks manifest JSON")
    parser.add_argument("--out_eval_csv", type=str, default="results/phase5_baseline_evaluation.csv", help="Output evaluation CSV")
    parser.add_argument("--out_summary_csv", type=str, default="results/phase5_baseline_summary.csv", help="Output summary CSV")
    parser.add_argument("--smoke", action="store_true", help="Run quick smoke test on 8 tasks per model (16 total)")
    parser.add_argument("--gui", action="store_true", help="Render PyBullet GUI")
    parser.add_argument("--episodes_per_task", type=int, default=1, help="Episodes per task")
    parser.add_argument("--episode_sec", type=float, default=6.0, help="Episode duration in seconds")
    args = parser.parse_args()

    protocol = EvaluationProtocol(
        config_path=args.config,
        manifest_path=args.manifest,
        gui=args.gui,
    )
    protocol.run_full_protocol(
        out_eval_csv=args.out_eval_csv,
        out_summary_csv=args.out_summary_csv,
        smoke_subset=args.smoke,
        episodes_per_task=args.episodes_per_task,
        episode_sec=args.episode_sec,
    )


if __name__ == "__main__":
    main()
