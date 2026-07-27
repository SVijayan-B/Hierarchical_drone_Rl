"""Disturbance Recovery Benchmark

Evaluate 7 UAV control stack configurations across 6 scenarios:
Scenario A: No Wind (Baseline)
Scenario B: Constant Wind (Constant 2.0N equivalent force)
Scenario C: Random Gusts (High frequency oscillatory wind)
Scenario D: Impulse Disturbance (Large force impulse at step 20)
Scenario E: Sensor Noise Attack (300% sensor noise std dev)
Scenario F: Motor Degradation (Rotors 0 and 1 degraded to 70% efficiency)
"""

import os
import time
import argparse
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import csv
import pybullet as p

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from hierarchical_drone.config.settings import ActionConfig, SensorConfig, SimConfig, TaskConfig
from hierarchical_drone.env.hierarchical_nav_env import HierarchicalNavEnv
from hierarchical_drone.benchmark.metrics import CompareConfig


def make_eval_env(
    gui=False,
    use_adaptive_scheduler=True,
    demo_guided_mode=True,
    use_mpc_layer=False,
    use_rl_gain_scheduler=False,
    use_adaptive_mpc=False,
    use_history=False,
    domain_randomization=False,
    telemetry_dir=None
):
    def _thunk():
        return HierarchicalNavEnv(
            sim=SimConfig(gui=gui),
            task=TaskConfig(),
            sensor_cfg=SensorConfig(),
            action_cfg=ActionConfig(),
            use_adaptive_scheduler=use_adaptive_scheduler,
            demo_guided_mode=demo_guided_mode,
            use_mpc_layer=use_mpc_layer,
            use_rl_gain_scheduler=use_rl_gain_scheduler,
            use_adaptive_mpc=use_adaptive_mpc,
            use_history=use_history,
            domain_randomization=domain_randomization,
            telemetry_dir=telemetry_dir
        )
    return _thunk


def main():
    parser = argparse.ArgumentParser(description="UAV Control Stack Disturbance Recovery Benchmark")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--model", type=str, default=None, help="Path to PPO MLP model zip")
    parser.add_argument("--vecnorm", type=str, default=None, help="Path to MLP vecnormalize pkl")
    parser.add_argument("--trans_model", type=str, default=None, help="Path to Transformer PPO model zip")
    parser.add_argument("--trans_vecnorm", type=str, default=None, help="Path to Transformer vecnormalize pkl")
    args = parser.parse_args()

    model_mlp_path = args.model
    vecnorm_mlp_path = args.vecnorm
    trans_model_path = args.trans_model
    trans_vecnorm_path = args.trans_vecnorm

    if not model_mlp_path:
        default_run = os.path.join("results_hierarchical", "run_20260513_112909")
        model_mlp_path = os.path.join(default_run, "final_model.zip")
        vecnorm_mlp_path = os.path.join(default_run, "vecnormalize.pkl")

        if not os.path.exists(model_mlp_path):
            # Auto-detect latest run in results_hierarchical
            runs = glob.glob(os.path.join("results_hierarchical", "run_*"))
            if runs:
                runs.sort()
                model_mlp_path = os.path.join(runs[-1], "final_model.zip")
                vecnorm_mlp_path = os.path.join(runs[-1], "vecnormalize.pkl")
            else:
                raise FileNotFoundError("Could not find baseline PPO MLP model.")

    trans_runs = glob.glob(os.path.join("results_hierarchical", "*trans*"))
    if not trans_model_path:
        # Auto-detect transformer model
        if trans_runs:
            trans_runs.sort()
            latest_trans_run = trans_runs[-1]
            best_model = os.path.join(latest_trans_run, "best", "best_model.zip")
            final_model = os.path.join(latest_trans_run, "final_model.zip")
            trans_model_path = best_model if os.path.exists(best_model) else final_model
            trans_vecnorm_path = os.path.join(latest_trans_run, "vecnormalize.pkl")

    # Load policies
    model_mlp = PPO.load(model_mlp_path)
    model_trans = PPO.load(trans_model_path) if trans_model_path else model_mlp
    trans_vecnorm_path = trans_vecnorm_path or vecnorm_mlp_path

    # Auto-detect PPO Gain Scheduler weights
    scheduler_pt_path = None
    if trans_runs:
        for run in reversed(trans_runs):
            for p_path in [os.path.join(run, "best", "scheduler_best.pt"), os.path.join(run, "scheduler_final.pt")]:
                if os.path.exists(p_path):
                    scheduler_pt_path = p_path
                    break
            if scheduler_pt_path:
                break

    # Configurations list
    configs = [
        ("pid_only", False, True, False, False, False, False, "PID-Only", False),
        ("rl_baseline", False, False, False, False, False, False, "PPO MLP + PID", False),
        ("rl_adaptive", True, False, False, False, False, False, "PPO MLP + Adaptive PID", False),
        ("rl_mpc_baseline", False, False, True, False, False, False, "PPO MLP + MPC + PID", False),
        ("rl_mpc_adaptive", True, False, True, False, False, False, "PPO MLP + MPC + Adaptive PID", False),
        ("trans_mpc_adaptive", True, False, True, False, False, True, "Transformer PPO + MPC + Adaptive PID", True),
    ]

    scenarios = [
        ("Scenario A", "No Wind"),
        ("Scenario B", "Constant Wind"),
        ("Scenario C", "Random Gusts"),
        ("Scenario D", "Impulse Disturbance"),
        ("Scenario E", "Sensor Noise Attack"),
        ("Scenario F", "Motor Degradation")
    ]

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("results", f"disturbance_plots_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 90)
    print("STARTING UAV CONTROL STACK DISTURBANCE RECOVERY BENCHMARK")
    print(f"Rounds: {args.rounds}, Seed: {args.seed}")
    print(f"PPO Scheduler checkpoint: {scheduler_pt_path}")
    print("=" * 90)

    # Initialize results structures
    # results[scenario][config] = list of ep metrics
    results = {s[0]: {c[0]: [] for c in configs} for s in scenarios}

    # Tracking error over time for plotting in Round 0
    # tracking_history[scenario][config] = list of tracking errors
    tracking_history = {s[0]: {} for s in scenarios}

    for s_id, s_name in scenarios:
        print(f"\n[SCENARIO] Running {s_id}: {s_name}")
        for key, use_scheduler, demo_mode, use_mpc, use_rl_gain, use_adaptive_mpc, use_history, disp, is_trans in configs:
            active_model = model_trans if is_trans else model_mlp
            active_vecnorm = trans_vecnorm_path if is_trans else vecnorm_mlp_path

            raw_env_fn = make_eval_env(
                gui=args.gui,
                use_adaptive_scheduler=use_scheduler,
                demo_guided_mode=demo_mode,
                use_mpc_layer=use_mpc,
                use_rl_gain_scheduler=use_rl_gain,
                use_adaptive_mpc=use_adaptive_mpc,
                use_history=use_history,
                domain_randomization=(s_id == "Scenario F"), # We explicitly customize DR below
                telemetry_dir=out_dir
            )

            vec_env = DummyVecEnv([raw_env_fn])
            vec_env = VecNormalize.load(active_vecnorm, vec_env)
            vec_env.training = False
            vec_env.norm_reward = False
            
            env = vec_env.envs[0]
            env.evaluation_mode = True

            if use_rl_gain and scheduler_pt_path:
                env.rl_gain_scheduler.load(scheduler_pt_path)

            ep_tracking_errors = []

            for r in range(args.rounds):
                # Reset environment with matched seed
                obs = vec_env.reset()
                env.set_telemetry_run_info(
                    run_name=timestamp,
                    config_name=f"scenario_{s_id.replace(' ', '_')}_{key}",
                    round_idx=r+1
                )
                
                # Apply Scenario Disturbance configuration on the raw environment
                if s_id == "Scenario A":
                    env.active_wind_disturbance = np.zeros(3, dtype=np.float32)
                elif s_id == "Scenario B":
                    env.active_wind_disturbance = np.array([2.0, 0.0, 0.0], dtype=np.float32) * 0.015
                    env.active_wind_freq = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                    env.wind_phase = np.array([np.pi/2, np.pi/2, np.pi/2], dtype=np.float32)
                elif s_id == "Scenario C":
                    env.active_wind_disturbance = np.array([1.5, 1.5, 0.0], dtype=np.float32) * 0.015
                    env.active_wind_freq = np.array([2.0, 2.0, 2.0], dtype=np.float32)
                elif s_id == "Scenario D":
                    env.active_wind_disturbance = np.zeros(3, dtype=np.float32)
                    # Monkey patch _apply_wind_disturbance to inject impulse at step 20
                    def patched_apply_wind(self):
                        force = np.zeros(3)
                        # step 20 corresponds to simulation time 2.0s, we apply impulse for 1 RL step (10 control cycles)
                        if self.step_count == 20:
                            force = np.array([4.5, 4.5, 0.0]) * 0.015
                        self.current_wind_magnitude = float(np.linalg.norm(force))
                        if self.current_wind_magnitude > 0:
                            p.applyExternalForce(
                                objectUniqueId=self.drone_id,
                                linkIndex=-1,
                                forceObj=force.tolist(),
                                posObj=[0.0, 0.0, 0.0],
                                flags=p.LINK_FRAME,
                                physicsClientId=self.client,
                            )
                    # Bind patch
                    import types
                    env._apply_wind_disturbance = types.MethodType(patched_apply_wind, env)
                elif s_id == "Scenario E":
                    env.active_wind_disturbance = np.zeros(3, dtype=np.float32)
                    env.imu.angle_noise_std = env.sensor_cfg.imu_angle_noise_std * 3.0
                    env.imu.rate_noise_std = env.sensor_cfg.imu_rate_noise_std * 3.0
                    env.imu.acc_noise_std = env.sensor_cfg.imu_acc_noise_std * 3.0
                    env.ultra.noise_std = env.sensor_cfg.ultrasonic_noise_std * 3.0
                elif s_id == "Scenario F":
                    env.active_wind_disturbance = np.zeros(3, dtype=np.float32)
                    env.domain_randomization = True
                    env.motor_efficiencies = np.array([0.70, 0.70, 1.0, 1.0], dtype=np.float32)

                done = False
                ep_reward = 0.0
                step_errors = []
                recovery_step = None
                pushed_off = False

                while not done:
                    action, _ = active_model.predict(obs, deterministic=True)
                    obs, reward, dones, infos = vec_env.step(action)
                    ep_reward += float(reward[0])
                    done = bool(dones[0])

                    info = infos[0]
                    # Track tracking error: dist ref is distance between drone and its tracking reference pos_ref
                    s_state = env.last_state
                    pos = s_state[0:3]
                    err = float(np.linalg.norm(pos - env.pos_ref))
                    step_errors.append(err)

                    # Scenario D recovery tracking
                    if s_id == "Scenario D":
                        if env.step_count == 20:
                            pushed_off = True
                        if pushed_off and env.step_count > 20 and err < 0.15 and recovery_step is None:
                            # Time to recover is (step_count - 20) * 0.1 seconds
                            recovery_step = env.step_count

                info = infos[0]
                success = float(info.get("success", 0.0))
                mean_err = info.get("mean_tracking_error", 0.0)
                tot_energy = info.get("total_energy", 0.0)
                smoothness = info.get("trajectory_smoothness", 0.0)
                avg_jerk = info.get("avg_jerk", 0.0)

                # Compute recovery time for Scenario D
                recovery_time = 60.0
                if s_id == "Scenario D":
                    if recovery_step is not None:
                        recovery_time = float((recovery_step - 20) * 0.1)
                    else:
                        recovery_time = 4.0  # Cap at max flight time since impulse

                results[s_id][key].append({
                    "success": success,
                    "reward": ep_reward,
                    "mean_error": mean_err,
                    "energy": tot_energy,
                    "smoothness": smoothness,
                    "avg_jerk": avg_jerk,
                    "recovery_time": recovery_time
                })

                if r == 0:
                    ep_tracking_errors = step_errors

            # Save tracking history for round 0 plotting
            tracking_history[s_id][key] = ep_tracking_errors
            vec_env.close()

    # Create CSV file with benchmark results
    csv_path = os.path.join(out_dir, "disturbance_benchmark_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Scenario", "Configuration", "Average_Reward", "Success_Rate", 
            "Mean_Tracking_Error_m", "Average_Jerk", "Total_Energy", "Recovery_Time_s"
        ])
        for s_id, s_name in scenarios:
            for key, _, _, _, _, _, _, disp, _ in configs:
                data_list = results[s_id][key]
                avg_rew = np.mean([d["reward"] for d in data_list])
                succ = np.mean([d["success"] for d in data_list])
                err = np.mean([d["mean_error"] for d in data_list])
                jerk = np.mean([d["avg_jerk"] for d in data_list])
                eng = np.mean([d["energy"] for d in data_list])
                rec = np.mean([d["recovery_time"] for d in data_list]) if s_id == "Scenario D" else 0.0

                writer.writerow([
                    s_name, disp, f"{avg_rew:.2f}", f"{succ:.1%}", 
                    f"{err:.4f}", f"{jerk:.2f}", f"{eng:.1f}", f"{rec:.2f}"
                ])
            writer.writerow([])

    print(f"\n[BENCHMARK] CSV metrics exported to {csv_path}")

    # Generate Performance Report
    report_path = os.path.join(out_dir, "disturbance_benchmark_report.txt")
    with open(report_path, "w") as f:
        def write_report(text):
            print(text)
            f.write(text + "\n")

        write_report("=" * 140)
        write_report("                                   UAV CONTROL STACK DISTURBANCE RECOVERY PERFORMANCE REPORT")
        write_report("=" * 140)
        write_report(f"Rounds per Scenario: {args.rounds}")
        write_report("-" * 140)

        for s_id, s_name in scenarios:
            write_report(f"\nScenario: {s_name}")
            write_report("-" * 140)
            write_report(
                f"{'Configuration':<50} | {'Success':<8} | {'Mean Error':<10} | {'Avg Jerk':<10} | {'Energy':<10} | {'Recovery Time':<13}"
            )
            write_report("-" * 140)
            for key, _, _, _, _, _, _, disp, _ in configs:
                d_list = results[s_id][key]
                succ = np.mean([d["success"] for d in d_list])
                err = np.mean([d["mean_error"] for d in d_list])
                jerk = np.mean([d["avg_jerk"] for d in d_list])
                eng = np.mean([d["energy"] for d in d_list])
                rec = np.mean([d["recovery_time"] for d in d_list]) if s_id == "Scenario D" else 0.0
                rec_str = f"{rec:.2f}s" if s_id == "Scenario D" else "N/A"

                write_report(
                    f"{disp:<50} | {succ:<8.1%} | {err:<10.4f} | {jerk:<10.2f} | {eng:<10.1f} | {rec_str:<13}"
                )
            write_report("-" * 140)

    print(f"[BENCHMARK] Performance report generated at {report_path}")

    # Generate Plot 1: Scenario D Tracking Error & Impulse Recovery Time
    plt.figure(figsize=(10, 6))
    colors = ['#7f7f7f', '#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c']
    for idx, (key, _, _, _, _, _, _, disp, _) in enumerate(configs):
        errors = tracking_history["Scenario D"].get(key, [])
        t_vec = np.arange(len(errors)) * 0.1
        plt.plot(t_vec, errors, color=colors[idx], label=disp)
    plt.axvline(x=2.0, color="k", linestyle="--", alpha=0.7, label="Impulse Applied (t=2.0s)")
    plt.axhline(y=0.15, color="r", linestyle=":", alpha=0.6, label="Recovery Threshold (0.15m)")
    plt.xlabel("Simulation Time (s)")
    plt.ylabel("Tracking Reference Error (m)")
    plt.title("UAV Impulse Disturbance Response & Recovery (Scenario D)")
    plt.grid(True)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "impulse_recovery_response.png"))
    plt.close()

    # Generate Plot 2: Success Rates across scenarios
    fig, ax = plt.subplots(figsize=(12, 7))
    x_indices = np.arange(len(scenarios))
    bar_width = 0.11

    for idx, (key, _, _, _, _, _, _, disp, _) in enumerate(configs):
        success_rates = [np.mean([d["success"] for d in results[s[0]][key]]) for s in scenarios]
        ax.bar(x_indices + idx * bar_width, success_rates, width=bar_width, color=colors[idx], label=disp)

    ax.set_xticks(x_indices + (len(configs) - 1) / 2 * bar_width)
    ax.set_xticklabels([s[1] for s in scenarios], rotation=15)
    ax.set_ylabel("Success Rate")
    ax.set_title("Configuration Robustness Success Rates Across Disturbance Scenarios")
    ax.grid(axis="y")
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "scenario_success_comparison.png"), bbox_inches="tight")
    plt.close()

    # Copy plots to the brain artifacts directory
    artifact_dir = "C:/Users/B Siddarth Vijayan/.gemini/antigravity-ide/brain/6ba40026-340c-4d66-a309-f317a44573a5"
    if os.path.exists(artifact_dir):
        import shutil
        try:
            shutil.copy(os.path.join(out_dir, "impulse_recovery_response.png"), os.path.join(artifact_dir, "impulse_recovery_response.png"))
            shutil.copy(os.path.join(out_dir, "scenario_success_comparison.png"), os.path.join(artifact_dir, "scenario_success_comparison.png"))
            print(f"[BENCHMARK] Copied disturbance plots to artifacts directory: {artifact_dir}")
        except Exception as e:
            print(f"[BENCHMARK] Could not copy files to artifact directory: {e}")


if __name__ == "__main__":
    main()
