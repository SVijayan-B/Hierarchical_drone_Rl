import os
import time
import numpy as np
import matplotlib
import csv

matplotlib.use("Agg")  # Safe headless execution
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from hierarchical_drone.config.settings import ActionConfig, SensorConfig, SimConfig, TaskConfig
from hierarchical_drone.env.hierarchical_nav_env import HierarchicalNavEnv
from hierarchical_drone.benchmark.metrics import CompareConfig


def make_eval_env(gui=False, use_adaptive_scheduler=True, demo_guided_mode=True, use_mpc_layer=False):
    def _thunk():
        return HierarchicalNavEnv(
            sim=SimConfig(gui=gui),
            task=TaskConfig(),
            sensor_cfg=SensorConfig(),
            action_cfg=ActionConfig(),
            use_adaptive_scheduler=use_adaptive_scheduler,
            demo_guided_mode=demo_guided_mode,
            use_mpc_layer=use_mpc_layer,
        )

    return _thunk


def run_compare(model_path, vecnorm_path, cfg: CompareConfig):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("results", f"metrics_plots_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print("STARTING 5-WAY HIERARCHICAL DRONE COMPARISON")
    print(f"Rounds: {cfg.rounds}, Seed base: {cfg.target_seed}, Model: {model_path}")
    print("=" * 60)

    model = PPO.load(model_path)

    # We will collect metrics for five configurations
    configs = [
        # (name, use_scheduler, demo_guided_mode, use_mpc, display_name)
        ("pid_only", False, True, False, "PID-Only"),
        ("rl_baseline", False, False, False, "RL + PID Baseline"),
        ("rl_adaptive", True, False, False, "RL + PID Adaptive"),
        ("rl_mpc_baseline", False, False, True, "RL + MPC + PID Baseline"),
        ("rl_mpc_adaptive", True, False, True, "RL + MPC + PID Adaptive"),
    ]

    runs_data = {}
    for key, _, _, _, disp in configs:
        runs_data[key] = {
            "rewards": [],
            "success_rate": [],
            "mean_err": [],
            "rmse_err": [],
            "overshoot": [],
            "settling_time": [],
            "smoothness": [],
            "control_effort": [],
            "wp_error": [],
        }

    # For plotting a single comparison run:
    dist_histories = {}
    gain_histories = {}
    time_histories = {}

    for key, use_scheduler, demo_mode, use_mpc, disp in configs:
        print(f"\nEvaluating configuration: {disp}")

        # Create environment
        raw_env_fn = make_eval_env(gui=cfg.gui, use_adaptive_scheduler=use_scheduler, demo_guided_mode=demo_mode, use_mpc_layer=use_mpc)
        vec_env = DummyVecEnv([raw_env_fn])
        vec_env = VecNormalize.load(vecnorm_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False

        for r in range(cfg.rounds):
            seed = cfg.target_seed + r
            obs = vec_env.reset()

            done = False
            ep_reward = 0.0

            dist_hist = []
            gain_hist = []

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, dones, infos = vec_env.step(action)
                ep_reward += float(reward[0])
                done = bool(dones[0])

                info = infos[0]
                dist_hist.append(info.get("distance", 0.0))
                gain_hist.append(info.get("gain_scale", 1.0))

                if cfg.gui:
                    time.sleep(0.01)

            info = infos[0]
            success = float(info.get("success", 0.0))
            mean_err = info.get("mean_tracking_error", 0.0)
            rmse_err = info.get("rmse_tracking_error", 0.0)
            overshoot = info.get("overshoot", 0.0)
            settling_time = info.get("settling_time", 40.0)
            smoothness = info.get("trajectory_smoothness", 0.0)
            control_effort = info.get("control_effort", 0.0)
            wp_error = info.get("waypoint_tracking_error", 0.0)

            runs_data[key]["rewards"].append(ep_reward)
            runs_data[key]["success_rate"].append(success)
            runs_data[key]["mean_err"].append(mean_err)
            runs_data[key]["rmse_err"].append(rmse_err)
            runs_data[key]["overshoot"].append(overshoot)
            runs_data[key]["settling_time"].append(settling_time)
            runs_data[key]["smoothness"].append(smoothness)
            runs_data[key]["control_effort"].append(control_effort)
            runs_data[key]["wp_error"].append(wp_error)

            print(
                f"Round {r+1}/{cfg.rounds}: Reward={ep_reward:.2f}, Success={success}, Tracking Error={mean_err:.3f}m, "
                f"Smoothness={smoothness:.1f}, Effort={control_effort:.4f}, WP Err={wp_error:.3f}m"
            )

            if r == 0:
                dist_histories[key] = dist_hist
                gain_histories[key] = gain_hist
                time_histories[key] = np.arange(len(dist_hist)) * 0.1

        vec_env.close()

    # Save to CSV
    csv_path = os.path.join(out_dir, "comparison_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Configuration", "Round", "Reward", "Success", "Mean_Tracking_Error_m", "RMSE_Tracking_Error_m", "Overshoot_m", "Settling_Time_s", "Trajectory_Smoothness", "Control_Effort", "Waypoint_Tracking_Error"
        ])

        for key, _, _, _, disp in configs:
            # Write individual rounds
            for r in range(cfg.rounds):
                writer.writerow([
                    disp,
                    r + 1,
                    f"{runs_data[key]['rewards'][r]:.2f}",
                    f"{runs_data[key]['success_rate'][r]:.1f}",
                    f"{runs_data[key]['mean_err'][r]:.4f}",
                    f"{runs_data[key]['rmse_err'][r]:.4f}",
                    f"{runs_data[key]['overshoot'][r]:.4f}",
                    f"{runs_data[key]['settling_time'][r]:.2f}",
                    f"{runs_data[key]['smoothness'][r]:.2f}",
                    f"{runs_data[key]['control_effort'][r]:.6f}",
                    f"{runs_data[key]['wp_error'][r]:.4f}",
                ])
            # Write averages
            writer.writerow([
                disp,
                "Average",
                f"{np.mean(runs_data[key]['rewards']):.2f}",
                f"{np.mean(runs_data[key]['success_rate']):.1%}",
                f"{np.mean(runs_data[key]['mean_err']):.4f}",
                f"{np.mean(runs_data[key]['rmse_err']):.4f}",
                f"{np.mean(runs_data[key]['overshoot']):.4f}",
                f"{np.mean(runs_data[key]['settling_time']):.2f}",
                f"{np.mean(runs_data[key]['smoothness']):.2f}",
                f"{np.mean(runs_data[key]['control_effort']):.6f}",
                f"{np.mean(runs_data[key]['wp_error']):.4f}",
            ])
            # Blank line spacer
            writer.writerow([])

    print(f"\nCSV results written to {csv_path}")

    # Process and print summaries
    summary_path = os.path.join(out_dir, "comparison_summary.txt")
    with open(summary_path, "w") as f:

        def write_and_print(text):
            print(text)
            f.write(text + "\n")

        write_and_print("=" * 115)
        write_and_print("                               5-WAY DRONE PERFORMANCE COMPARISON REPORT")
        write_and_print("=" * 115)
        write_and_print(f"Model: {model_path}")
        write_and_print(f"Evaluated over {cfg.rounds} rounds with matched seeds.")
        write_and_print("-" * 115)
        write_and_print(f"{'Metric':<28} | {'PID-Only':<12} | {'RL Baseline':<12} | {'RL Adaptive':<12} | {'RL+MPC Base':<12} | {'RL+MPC Adap':<12}")
        write_and_print("-" * 115)

        metrics_to_print = [
            ("Success Rate", "success_rate", "{:.1%}"),
            ("Average Reward", "rewards", "{:.2f}"),
            ("Mean Tracking Error (m)", "mean_err", "{:.4f}"),
            ("RMSE Tracking Error (m)", "rmse_err", "{:.4f}"),
            ("Average Overshoot (m)", "overshoot", "{:.4f}"),
            ("Average Settling Time (s)", "settling_time", "{:.2f}"),
            ("Trajectory Smoothness", "smoothness", "{:.2f}"),
            ("Control Effort", "control_effort", "{:.6f}"),
            ("Waypoint Tracking Error (m)", "wp_error", "{:.4f}"),
        ]

        for name, key, fmt in metrics_to_print:
            pid_val = np.mean(runs_data["pid_only"][key])
            base_val = np.mean(runs_data["rl_baseline"][key])
            adap_val = np.mean(runs_data["rl_adaptive"][key])
            mpc_base_val = np.mean(runs_data["rl_mpc_baseline"][key])
            mpc_adap_val = np.mean(runs_data["rl_mpc_adaptive"][key])

            pid_str = fmt.format(pid_val)
            base_str = fmt.format(base_val)
            adap_str = fmt.format(adap_val)
            mpc_base_str = fmt.format(mpc_base_val)
            mpc_adap_str = fmt.format(mpc_adap_val)

            write_and_print(f"{name:<28} | {pid_str:<12} | {base_str:<12} | {adap_str:<12} | {mpc_base_str:<12} | {mpc_adap_str:<12}")

        write_and_print("=" * 115)

    print(f"Summary report written to {summary_path}")

    # Plotting
    # Plot 1: Target Distance over time comparison for round 0
    plt.figure(figsize=(10, 5))
    if "pid_only" in dist_histories:
        plt.plot(time_histories["pid_only"][: len(dist_histories["pid_only"])], dist_histories["pid_only"], "k:", label="PID-Only (No RL)")
    if "rl_baseline" in dist_histories:
        plt.plot(time_histories["rl_baseline"][: len(dist_histories["rl_baseline"])], dist_histories["rl_baseline"], "r--", label="RL + PID Baseline")
    if "rl_adaptive" in dist_histories:
        plt.plot(time_histories["rl_adaptive"][: len(dist_histories["rl_adaptive"])], dist_histories["rl_adaptive"], "b-", label="RL + PID Adaptive")
    if "rl_mpc_baseline" in dist_histories:
        plt.plot(time_histories["rl_mpc_baseline"][: len(dist_histories["rl_mpc_baseline"])], dist_histories["rl_mpc_baseline"], "m-.", label="RL + MPC + PID Baseline")
    if "rl_mpc_adaptive" in dist_histories:
        plt.plot(time_histories["rl_mpc_adaptive"][: len(dist_histories["rl_mpc_adaptive"])], dist_histories["rl_mpc_adaptive"], "g-", label="RL + MPC + PID Adaptive")
    plt.axhline(y=0.60, color="g", linestyle="-.", label="Target Threshold (0.60m)")
    plt.xlabel("Simulation Time (s)")
    plt.ylabel("Distance to Target (m)")
    plt.title("Target Distance Tracking over Time (Round 1 Comparison)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "tracking_comparison.png"))
    plt.close()

    # Plot 2: Gain Scale over time for round 0
    plt.figure(figsize=(10, 4))
    if "rl_adaptive" in gain_histories:
        plt.plot(time_histories["rl_adaptive"][: len(gain_histories["rl_adaptive"])], gain_histories["rl_adaptive"], "b--", label="RL + PID Adaptive")
    if "rl_mpc_adaptive" in gain_histories:
        plt.plot(time_histories["rl_mpc_adaptive"][: len(gain_histories["rl_mpc_adaptive"])], gain_histories["rl_mpc_adaptive"], "g-", label="RL + MPC + PID Adaptive")
    plt.xlabel("Simulation Time (s)")
    plt.ylabel("Gain Scale (multiplier)")
    plt.title("Adaptive PID Gain Scale Modulation (Round 1)")
    plt.axhline(y=1.0, color="r", linestyle="--", label="Baseline (1.0)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "gain_scale_history.png"))
    plt.close()

    # Plot 3: Unified bar charts in a 2x3 grid
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    categories = ["PID-Only", "RL Baseline", "RL Adaptive", "RL+MPC Base", "RL+MPC Adap"]
    colors = ["grey", "red", "blue", "magenta", "green"]

    def plot_bar(ax, metric_key, title, ylabel):
        vals = [np.mean(runs_data[k][metric_key]) for k in ["pid_only", "rl_baseline", "rl_adaptive", "rl_mpc_baseline", "rl_mpc_adaptive"]]
        ax.bar(categories, vals, color=colors, width=0.4)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y")
        ax.tick_params(axis="x", labelrotation=15)

    plot_bar(axes[0, 0], "mean_err", "Mean Tracking Error", "Error (m)")
    plot_bar(axes[0, 1], "settling_time", "Average Settling Time", "Time (s)")
    plot_bar(axes[0, 2], "rewards", "Average Episode Reward", "Reward")
    plot_bar(axes[1, 0], "smoothness", "Trajectory Smoothness (RMS Jerk)", "Smoothness (lower is better)")
    plot_bar(axes[1, 1], "control_effort", "Control Effort", "Effort (lower is better)")
    plot_bar(axes[1, 2], "wp_error", "Waypoint Tracking Error", "Error (m)")

    plt.suptitle("Performance Metrics Unified Comparison (5-Way)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(out_dir, "metrics_comparison.png"))
    plt.close()

    print(
        f"Unified charts saved to {out_dir}/tracking_comparison.png, {out_dir}/gain_scale_history.png, and {out_dir}/metrics_comparison.png"
    )
