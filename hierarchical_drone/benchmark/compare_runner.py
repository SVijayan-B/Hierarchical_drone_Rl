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


def make_eval_env(gui=False, use_adaptive_scheduler=True, demo_guided_mode=True):
    def _thunk():
        return HierarchicalNavEnv(
            sim=SimConfig(gui=gui),
            task=TaskConfig(),
            sensor_cfg=SensorConfig(),
            action_cfg=ActionConfig(),
            use_adaptive_scheduler=use_adaptive_scheduler,
            demo_guided_mode=demo_guided_mode,
        )

    return _thunk


def run_compare(model_path, vecnorm_path, cfg: CompareConfig):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("results", f"metrics_plots_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print("STARTING 3-WAY HIERARCHICAL DRONE COMPARISON")
    print(f"Rounds: {cfg.rounds}, Seed base: {cfg.target_seed}, Model: {model_path}")
    print("=" * 60)

    model = PPO.load(model_path)

    # We will collect metrics for three configurations
    configs = [
        # (name, use_scheduler, demo_guided_mode, display_name)
        ("pid_only", False, True, "PID-Only"),
        ("rl_baseline", False, False, "RL + PID Baseline"),
        ("rl_adaptive", True, False, "RL + PID Adaptive"),
    ]

    runs_data = {}
    for key, _, _, disp in configs:
        runs_data[key] = {
            "rewards": [],
            "success_rate": [],
            "mean_err": [],
            "rmse_err": [],
            "overshoot": [],
            "settling_time": [],
        }

    # For plotting a single comparison run:
    dist_histories = {}
    gain_histories = {}
    time_histories = {}

    for key, use_scheduler, demo_mode, disp in configs:
        print(f"\nEvaluating configuration: {disp}")

        # Create environment
        raw_env_fn = make_eval_env(gui=cfg.gui, use_adaptive_scheduler=use_scheduler, demo_guided_mode=demo_mode)
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

            runs_data[key]["rewards"].append(ep_reward)
            runs_data[key]["success_rate"].append(success)
            runs_data[key]["mean_err"].append(mean_err)
            runs_data[key]["rmse_err"].append(rmse_err)
            runs_data[key]["overshoot"].append(overshoot)
            runs_data[key]["settling_time"].append(settling_time)

            print(
                f"Round {r+1}/{cfg.rounds}: Reward={ep_reward:.2f}, Success={success}, Tracking Error={mean_err:.3f}m, Overshoot={overshoot:.3f}m, Settling Time={settling_time:.2f}s"
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
            "Configuration", "Round", "Reward", "Success", "Mean_Tracking_Error_m", "RMSE_Tracking_Error_m", "Overshoot_m", "Settling_Time_s"
        ])

        for key, _, _, disp in configs:
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

        write_and_print("=" * 70)
        write_and_print("           3-WAY DRONE PERFORMANCE COMPARISON REPORT")
        write_and_print("=" * 70)
        write_and_print(f"Model: {model_path}")
        write_and_print(f"Evaluated over {cfg.rounds} rounds with matched seeds.")
        write_and_print("-" * 70)
        write_and_print(f"{'Metric':<25} | {'PID-Only':<12} | {'RL Baseline':<12} | {'RL Adaptive':<12}")
        write_and_print("-" * 70)

        metrics_to_print = [
            ("Success Rate", "success_rate", "{:.1%}"),
            ("Average Reward", "rewards", "{:.2f}"),
            ("Mean Tracking Error (m)", "mean_err", "{:.4f}"),
            ("RMSE Tracking Error (m)", "rmse_err", "{:.4f}"),
            ("Average Overshoot (m)", "overshoot", "{:.4f}"),
            ("Average Settling Time (s)", "settling_time", "{:.2f}"),
        ]

        for name, key, fmt in metrics_to_print:
            pid_val = np.mean(runs_data["pid_only"][key])
            base_val = np.mean(runs_data["rl_baseline"][key])
            adap_val = np.mean(runs_data["rl_adaptive"][key])

            pid_str = fmt.format(pid_val)
            base_str = fmt.format(base_val)
            adap_str = fmt.format(adap_val)

            write_and_print(f"{name:<25} | {pid_str:<12} | {base_str:<12} | {adap_str:<12}")

        write_and_print("=" * 70)

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
        plt.plot(time_histories["rl_adaptive"][: len(gain_histories["rl_adaptive"])], gain_histories["rl_adaptive"], "g-", label="RL + PID Adaptive Gain Scale")
    plt.xlabel("Simulation Time (s)")
    plt.ylabel("Gain Scale (multiplier)")
    plt.title("Adaptive PID Gain Scale Modulation (Round 1)")
    plt.axhline(y=1.0, color="r", linestyle="--", label="Baseline (1.0)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "gain_scale_history.png"))
    plt.close()

    # Plot 3: Unified bar charts in a single sheet
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    categories = ["PID-Only", "RL Baseline", "RL Adaptive"]
    colors = ["grey", "red", "blue"]

    # Mean Tracking Error
    axes[0, 0].bar(categories, [np.mean(runs_data["pid_only"]["mean_err"]), np.mean(runs_data["rl_baseline"]["mean_err"]), np.mean(runs_data["rl_adaptive"]["mean_err"])], color=colors, width=0.4)
    axes[0, 0].set_ylabel("Error (m)")
    axes[0, 0].set_title("Mean Tracking Error")
    axes[0, 0].grid(axis="y")

    # Average Overshoot
    axes[0, 1].bar(categories, [np.mean(runs_data["pid_only"]["overshoot"]), np.mean(runs_data["rl_baseline"]["overshoot"]), np.mean(runs_data["rl_adaptive"]["overshoot"])], color=colors, width=0.4)
    axes[0, 1].set_ylabel("Overshoot (m)")
    axes[0, 1].set_title("Average Overshoot")
    axes[0, 1].grid(axis="y")

    # Average Settling Time
    axes[1, 0].bar(categories, [np.mean(runs_data["pid_only"]["settling_time"]), np.mean(runs_data["rl_baseline"]["settling_time"]), np.mean(runs_data["rl_adaptive"]["settling_time"])], color=colors, width=0.4)
    axes[1, 0].set_ylabel("Time (s)")
    axes[1, 0].set_title("Average Settling Time")
    axes[1, 0].grid(axis="y")

    # Average Reward
    axes[1, 1].bar(categories, [np.mean(runs_data["pid_only"]["rewards"]), np.mean(runs_data["rl_baseline"]["rewards"]), np.mean(runs_data["rl_adaptive"]["rewards"])], color=colors, width=0.4)
    axes[1, 1].set_ylabel("Reward")
    axes[1, 1].set_title("Average Episode Reward")
    axes[1, 1].grid(axis="y")

    plt.suptitle("Performance Metrics Unified Comparison", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(out_dir, "metrics_comparison.png"))
    plt.close()

    print(
        f"Unified charts saved to {out_dir}/tracking_comparison.png, {out_dir}/gain_scale_history.png, and {out_dir}/metrics_comparison.png"
    )
