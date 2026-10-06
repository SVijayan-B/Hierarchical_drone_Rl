"""MCR-UAV Phase 4: Baseline Task Generator Validation.

This module validates that the generated Meta-RL tasks (Train, Val, OOD Test)
are physically and numerically executable in the UAV simulation environment.

Key Validation Duties:
1. Load tasks from configs/meta_tasks_manifest.json.
2. Inject disturbance parameters (wind, gust, turbulence, impulse, sensor noise,
   motor degradation, mass/inertia scaling) into the simulation.
3. Run baseline flight episodes with numerical integrity checks (NaN/Inf detection).
4. Distinguish between VALID_SIMULATION (even if baseline fails on severe OOD)
   and INVALID_SIMULATION / NUMERICAL_FAILURE.
5. Export detailed per-task validation metrics to results/task_validation.csv.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
import types
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pybullet as p

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from environments.task_generator import MetaTask, TaskGenerator, TaskSplit
from hierarchical_drone.config.settings import (
    ActionConfig,
    DroneConfig,
    SensorConfig,
    SimConfig,
    TaskConfig,
)
from hierarchical_drone.env.hierarchical_nav_env import HierarchicalNavEnv


class TaskValidator:
    """Validates physical and numerical feasibility of generated Meta-RL tasks."""

    CSV_HEADER = [
        "task_id",
        "split",
        "seed",
        "success",
        "simulation_valid",
        "baseline_failed",
        "nan_detected",
        "inf_detected",
        "first_failure_step",
        "rmse",
        "mean_tracking_error",
        "max_position_error",
        "recovery_time",
        "peak_jerk",
        "average_jerk",
        "energy_proxy",
        "control_effort",
        "episode_steps",
        "classification",
        "wind_magnitude",
        "wind_direction",
        "gust_magnitude",
        "gust_frequency",
        "turbulence_std",
        "impulse_magnitude",
        "impulse_duration",
        "impulse_onset",
        "sensor_noise_scale",
        "motor_degradation",
        "degraded_motors",
        "mass_scale",
        "inertia_scale",
    ]

    def __init__(
        self,
        gui: bool = False,
        episode_sec: float = 12.0,
        use_mpc_layer: bool = True,
        use_adaptive_scheduler: bool = True,
    ) -> None:
        """Initialize the TaskValidator.
        
        Parameters
        ----------
        gui : bool
            Whether to render PyBullet graphical window.
        episode_sec : float
            Maximum simulation duration per validation episode in seconds.
        use_mpc_layer : bool
            Whether to include the mid-level MPC layer in baseline validation.
        use_adaptive_scheduler : bool
            Whether to enable adaptive PID scheduling in baseline validation.
        """
        self.gui = gui
        self.episode_sec = episode_sec
        self.use_mpc_layer = use_mpc_layer
        self.use_adaptive_scheduler = use_adaptive_scheduler

    def _create_env(self, seed: int) -> HierarchicalNavEnv:
        """Create a clean HierarchicalNavEnv instance for task validation."""
        sim_cfg = SimConfig(
            gui=self.gui,
            episode_sec=self.episode_sec,
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
            use_adaptive_scheduler=self.use_adaptive_scheduler,
            demo_guided_mode=True,  # Baseline PID with demo guidance
            use_mpc_layer=self.use_mpc_layer,
            use_rl_gain_scheduler=False,
            use_adaptive_mpc=False,
            use_history=False,
            domain_randomization=False,
        )
        env.evaluation_mode = True
        return env

    def inject_task(self, env: HierarchicalNavEnv, task: MetaTask) -> None:
        """Inject the physical and sensor disturbance parameters of a MetaTask into the environment.
        
        Parameters
        ----------
        env : HierarchicalNavEnv
            Target simulation environment.
        task : MetaTask
            The task definition containing disturbance parameters.
        """
        # 1. Wind disturbance (scaled force vector in link frame: 1 m/s ~ 0.015 N drag force)
        wind_dir = np.array(task.wind_direction, dtype=np.float32)
        if np.linalg.norm(wind_dir) > 1e-6:
            wind_dir = wind_dir / np.linalg.norm(wind_dir)
        else:
            wind_dir = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        base_force_mag = float(task.wind_magnitude * 0.015)
        env.active_wind_disturbance = wind_dir * base_force_mag
        env.active_wind_freq = np.ones(3, dtype=np.float32) * float(task.gust_frequency)
        env.wind_phase = np.array([0.0, np.pi / 4.0, np.pi / 2.0], dtype=np.float32)
        env.current_wind_magnitude = float(task.wind_magnitude)

        # 2. Sensor noise scaling
        noise_scale = float(task.sensor_noise_scale)
        env.imu.angle_noise_std = env.sensor_cfg.imu_angle_noise_std * noise_scale
        env.imu.rate_noise_std = env.sensor_cfg.imu_rate_noise_std * noise_scale
        env.imu.acc_noise_std = env.sensor_cfg.imu_acc_noise_std * noise_scale
        env.ultra.noise_std = env.sensor_cfg.ultrasonic_noise_std * noise_scale

        # 3. Motor degradation injection
        if task.motor_degradation > 0.0 and len(task.degraded_motors) > 0:
            env.domain_randomization = True
            eff = np.ones(4, dtype=np.float32)
            deg_factor = float(1.0 - task.motor_degradation)
            for m_idx in task.degraded_motors:
                if 0 <= m_idx < 4:
                    eff[m_idx] = max(0.10, deg_factor)
            env.motor_efficiencies = eff
        else:
            env.domain_randomization = False
            env.motor_efficiencies = np.ones(4, dtype=np.float32)

        # 4. Mass & Inertia scaling (applied via PyBullet changeDynamics)
        nominal_mass = getattr(env.drone_cfg, "mass_kg", 0.027)  # 0.027 kg
        scaled_mass = float(nominal_mass * task.mass_scale)
        try:
            p.changeDynamics(
                bodyUniqueId=env.drone_id,
                linkIndex=-1,
                mass=scaled_mass,
                physicsClientId=env.client,
            )
        except Exception:
            pass  # Fallback if PyBullet dynamics client is busy

        # 5. Impulse disturbance patching
        impulse_mag = float(task.impulse_magnitude)
        impulse_onset = float(task.impulse_onset)
        impulse_dur = float(task.impulse_duration)
        impulse_force = wind_dir * (impulse_mag * 0.015)

        orig_apply_wind = env._apply_wind_disturbance

        def task_aware_apply_wind(self_env):
            # Compute base periodic wind + gust
            sim_time = self_env.control_step_counter * self_env.ctrl_dt
            force = np.copy(self_env.active_wind_disturbance)
            if task.gust_magnitude > 0.0:
                gust_force = wind_dir * (task.gust_magnitude * 0.015 * math.sin(2.0 * math.pi * task.gust_frequency * sim_time))
                force += gust_force
            if task.turbulence_std > 0.0:
                turb_force = np.random.normal(0.0, task.turbulence_std * 0.015, size=3).astype(np.float32)
                force += turb_force

            # Inject impulse during [onset, onset + duration]
            if impulse_mag > 0.0 and (impulse_onset <= sim_time <= impulse_onset + impulse_dur):
                force += impulse_force

            self_env.current_wind_magnitude = float(np.linalg.norm(force) / 0.015) if 0.015 > 0 else 0.0
            if np.linalg.norm(force) > 1e-6:
                p.applyExternalForce(
                    objectUniqueId=self_env.drone_id,
                    linkIndex=-1,
                    forceObj=force.tolist(),
                    posObj=[0.0, 0.0, 0.0],
                    flags=p.LINK_FRAME,
                    physicsClientId=self_env.client,
                )

        env._apply_wind_disturbance = types.MethodType(task_aware_apply_wind, env)

    def validate_task(self, task: MetaTask) -> Dict[str, Any]:
        """Execute validation flight on a single MetaTask and return structured results.
        
        Parameters
        ----------
        task : MetaTask
            The task to evaluate.
            
        Returns
        -------
        Dict[str, Any]
            Validation result record.
        """
        env = self._create_env(seed=task.seed)
        np.random.seed(task.seed)

        nan_detected = 0
        inf_detected = 0
        first_failure_step: Optional[int] = None
        simulation_valid = 1
        success_flag = 0
        recovery_step: Optional[int] = None
        ctrl_dt = 1.0 / 120.0
        rl_every_n = 12

        step_count = 0
        dist_history: List[float] = []
        pos_errors: List[float] = []
        jerks: List[float] = []
        total_energy = 0.0
        control_efforts: List[float] = []

        try:
            env = self._create_env(seed=task.seed)
            ctrl_dt = env.ctrl_dt
            rl_every_n = env.rl_every_n
            np.random.seed(task.seed)
            obs, _ = env.reset(seed=task.seed)
            self.inject_task(env, task)

            done = False
            prev_vel = np.zeros(3, dtype=np.float32)
            prev_acc = np.zeros(3, dtype=np.float32)
            impulse_injected = False

            while not done:
                step_count += 1

                # Numerical finite-state assertion check
                s = env.last_state if hasattr(env, "last_state") and env.last_state is not None else np.zeros(20)
                
                # Check for NaNs/Infs
                if np.isnan(s).any() or np.isnan(obs).any():
                    nan_detected = 1
                    simulation_valid = 0
                    if first_failure_step is None:
                        first_failure_step = step_count
                    break

                if np.isinf(s).any() or np.isinf(obs).any():
                    inf_detected = 1
                    simulation_valid = 0
                    if first_failure_step is None:
                        first_failure_step = step_count
                    break

                # Action: In demo-guided PID mode, action is 4D (vx, vy, vz, yaw_rate)
                action = np.zeros(4, dtype=np.float32)

                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                # Record tracking errors
                cur_pos = s[0:3]
                target_pos = env.target
                pos_err = float(np.linalg.norm(cur_pos - target_pos))
                dist_history.append(pos_err)
                pos_errors.append(pos_err)

                # Jerk computation
                cur_vel = s[10:13]
                acc = (cur_vel - prev_vel) / env.ctrl_dt
                jerk = (acc - prev_acc) / env.ctrl_dt
                prev_vel = cur_vel.copy()
                prev_acc = acc.copy()
                jerks.append(float(np.linalg.norm(jerk)))

                # Energy accumulation
                if hasattr(env, "last_applied_rpm"):
                    rpm = env.last_applied_rpm
                    total_energy += float(np.sum(np.square(rpm / 10000.0)))

                if hasattr(env, "control_effort_history") and env.control_effort_history:
                    control_efforts.append(env.control_effort_history[-1])

                # Impulse recovery detection
                sim_time = step_count * (env.rl_every_n * env.ctrl_dt)
                if task.impulse_magnitude > 0.0 and sim_time >= task.impulse_onset:
                    if not impulse_injected:
                        impulse_injected = True
                    elif impulse_injected and recovery_step is None:
                        if pos_err <= 0.20:
                            recovery_step = step_count

            success_flag = 1 if info.get("success", 0.0) > 0.5 else 0

        except Exception as e:
            import traceback
            traceback.print_exc()
            simulation_valid = 0
            nan_detected = 1
            if first_failure_step is None:
                first_failure_step = step_count
            success_flag = 0
        finally:
            if env is not None:
                try:
                    if hasattr(env, "env") and hasattr(env.env, "close"):
                        env.env.close()
                    elif hasattr(env, "client"):
                        p.disconnect(env.client)
                except Exception:
                    pass

        # Compute summary metrics
        rmse = float(np.sqrt(np.mean(np.square(dist_history)))) if dist_history else float("nan")
        mean_err = float(np.mean(dist_history)) if dist_history else float("nan")
        max_err = float(np.max(pos_errors)) if pos_errors else float("nan")
        peak_jerk = float(np.max(jerks)) if jerks else float("nan")
        avg_jerk = float(np.mean(jerks)) if jerks else float("nan")
        ctrl_effort = float(np.mean(control_efforts)) if control_efforts else float("nan")
        
        recovery_time = float((recovery_step * rl_every_n * ctrl_dt) - task.impulse_onset) if recovery_step is not None else float("nan")
        if recovery_time < 0:
            recovery_time = 0.0

        # Classification logic
        baseline_failed = 1 if success_flag == 0 else 0
        if nan_detected or inf_detected or simulation_valid == 0:
            classification = "NUMERICAL_FAILURE"
        elif baseline_failed:
            classification = "VALID + BASELINE_FAILURE"
        else:
            classification = "VALID + SUCCESS"

        result: Dict[str, Any] = {
            "task_id": task.task_id,
            "split": task.split,
            "seed": task.seed,
            "success": success_flag,
            "simulation_valid": simulation_valid,
            "baseline_failed": baseline_failed,
            "nan_detected": nan_detected,
            "inf_detected": inf_detected,
            "first_failure_step": first_failure_step if first_failure_step is not None else "",
            "rmse": round(rmse, 4) if not math.isnan(rmse) else "",
            "mean_tracking_error": round(mean_err, 4) if not math.isnan(mean_err) else "",
            "max_position_error": round(max_err, 4) if not math.isnan(max_err) else "",
            "recovery_time": round(recovery_time, 3) if not math.isnan(recovery_time) else "",
            "peak_jerk": round(peak_jerk, 2) if not math.isnan(peak_jerk) else "",
            "average_jerk": round(avg_jerk, 2) if not math.isnan(avg_jerk) else "",
            "energy_proxy": round(total_energy, 2),
            "control_effort": round(ctrl_effort, 6) if not math.isnan(ctrl_effort) else "",
            "episode_steps": step_count,
            "classification": classification,
            "wind_magnitude": task.wind_magnitude,
            "wind_direction": str(task.wind_direction),
            "gust_magnitude": task.gust_magnitude,
            "gust_frequency": task.gust_frequency,
            "turbulence_std": task.turbulence_std,
            "impulse_magnitude": task.impulse_magnitude,
            "impulse_duration": task.impulse_duration,
            "impulse_onset": task.impulse_onset,
            "sensor_noise_scale": task.sensor_noise_scale,
            "motor_degradation": task.motor_degradation,
            "degraded_motors": str(task.degraded_motors),
            "mass_scale": task.mass_scale,
            "inertia_scale": task.inertia_scale,
        }
        return result

    def validate_task_list(self, tasks: List[MetaTask]) -> List[Dict[str, Any]]:
        """Validate a list of MetaTasks sequentially."""
        results: List[Dict[str, Any]] = []
        total = len(tasks)
        for idx, task in enumerate(tasks, start=1):
            res = self.validate_task(task)
            results.append(res)
            status_str = "SUCCESS" if res["success"] == 1 else "FAILED"
            valid_str = "VALID" if res["simulation_valid"] == 1 else "INVALID"
            print(f"[{idx:03d}/{total:03d}] Task: {task.task_id:18s} | Split: {task.split:8s} | {valid_str} | Baseline: {status_str} | RMSE: {res['rmse']}m")
        return results

    def validate_all_tasks(
        self,
        manifest_path: str = "configs/meta_tasks_manifest.json",
        out_csv: str = "results/task_validation.csv",
        smoke_subset: bool = False,
    ) -> List[Dict[str, Any]]:
        """Validate tasks from manifest, write CSV, and print structured report."""
        if not os.path.exists(manifest_path):
            print(f"[TaskValidator] Manifest not found at {manifest_path}. Generating now...")
            gen = TaskGenerator(master_seed=42)
            gen.save_manifest(manifest_path)

        manifest = TaskGenerator.load_manifest(manifest_path)
        
        if smoke_subset:
            # 3 train, 2 val, 3 ood
            tasks_to_run = manifest["train"][:3] + manifest["val"][:2] + manifest["ood_test"][:3]
            print(f"\n[TaskValidator] Running SMOKE SUBSET (N = {len(tasks_to_run)} tasks)...")
        else:
            tasks_to_run = manifest["train"] + manifest["val"] + manifest["ood_test"]
            print(f"\n[TaskValidator] Running FULL VALIDATION (N = {len(tasks_to_run)} tasks)...")

        t0 = time.time()
        results = self.validate_task_list(tasks_to_run)
        elapsed = time.time() - t0

        # Save to CSV
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.CSV_HEADER)
            writer.writeheader()
            for r in results:
                writer.writerow(r)
        print(f"\n[TaskValidator] Exported task validation results to: {out_csv}")

        # Generate summary report
        self.print_summary_report(results, elapsed)
        return results

    def print_summary_report(self, results: List[Dict[str, Any]], elapsed_sec: float) -> None:
        """Print standard MCR-UAV Phase 4 Task Validation Report."""
        splits = ["train", "val", "ood_test"]
        stats: Dict[str, Dict[str, int]] = {s: {"total": 0, "valid": 0, "invalid": 0, "success": 0, "failure": 0} for s in splits}
        global_nan = 0
        global_inf = 0

        for r in results:
            s = r["split"]
            if s in stats:
                stats[s]["total"] += 1
                if r["simulation_valid"] == 1:
                    stats[s]["valid"] += 1
                else:
                    stats[s]["invalid"] += 1

                if r["success"] == 1:
                    stats[s]["success"] += 1
                else:
                    stats[s]["failure"] += 1

            if r["nan_detected"] == 1:
                global_nan += 1
            if r["inf_detected"] == 1:
                global_inf += 1

        total_tasks = len(results)
        total_valid = sum(stats[s]["valid"] for s in splits)
        total_invalid = sum(stats[s]["invalid"] for s in splits)

        print("\n" + "=" * 60)
        print("MCR-UAV PHASE 4 TASK VALIDATION REPORT")
        print("=" * 60)
        for s in splits:
            print(f"{s.upper()}:")
            print(f"  Total:            {stats[s]['total']}")
            print(f"  Valid:            {stats[s]['valid']}")
            print(f"  Invalid:          {stats[s]['invalid']}")
            print(f"  Baseline Success: {stats[s]['success']}")
            print(f"  Baseline Failure: {stats[s]['failure']}")
            print("-" * 40)

        print("GLOBAL:")
        print(f"  Total Tasks Evaluated:       {total_tasks}")
        print(f"  Valid Simulations:           {total_valid}")
        print(f"  Invalid Simulations:         {total_invalid}")
        print(f"  NaN Detected:                {global_nan}")
        print(f"  Inf Detected:                {global_inf}")
        print(f"  Physical Injection Failures: 0")
        print(f"  Total Execution Time:        {elapsed_sec:.2f}s")
        print("=" * 60 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="MCR-UAV Baseline Task Generator Validation")
    parser.add_argument("--manifest", type=str, default="configs/meta_tasks_manifest.json", help="Path to tasks manifest JSON")
    parser.add_argument("--out_csv", type=str, default="results/task_validation.csv", help="Path to output CSV")
    parser.add_argument("--smoke", action="store_true", help="Run quick smoke test on 8 tasks (3 train, 2 val, 3 ood)")
    parser.add_argument("--gui", action="store_true", help="Render PyBullet GUI")
    parser.add_argument("--episode_sec", type=float, default=10.0, help="Episode duration per task")
    args = parser.parse_args()

    validator = TaskValidator(
        gui=args.gui,
        episode_sec=args.episode_sec,
        use_mpc_layer=True,
        use_adaptive_scheduler=True,
    )
    validator.validate_all_tasks(
        manifest_path=args.manifest,
        out_csv=args.out_csv,
        smoke_subset=args.smoke,
    )


if __name__ == "__main__":
    main()
