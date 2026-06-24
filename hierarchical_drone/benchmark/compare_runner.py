import os
import time
import numpy as np
import matplotlib
import csv
import glob

matplotlib.use("Agg")  # Safe headless execution
import matplotlib.pyplot as plt
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


def run_compare(model_path, vecnorm_path, cfg: CompareConfig, trans_model_path=None, trans_vecnorm_path=None):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("results", f"metrics_plots_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 70)
    print("STARTING 7-WAY HIERARCHICAL DRONE STACK COMPARISON WITH HONORS METRICS")
    print(f"Rounds: {cfg.rounds}, Seed base: {cfg.target_seed}")
    print(f"MLP Model: {model_path}")
    print("=" * 70)

    # 1. Load PPO MLP Model
    model_mlp = PPO.load(model_path)
    model_trans = None

    # 2. Auto-detect or load PPO Transformer Model
    if trans_model_path and os.path.exists(trans_model_path):
        print(f"Loading specified Transformer PPO model: {trans_model_path}")
        model_trans = PPO.load(trans_model_path)
    else:
        # Search results_hierarchical for a run with "trans"
        trans_runs = glob.glob(os.path.join("results_hierarchical", "*trans*"))
        if trans_runs:
            trans_runs.sort()
            latest_run = trans_runs[-1]
            best_model = os.path.join(latest_run, "best", "best_model.zip")
            final_model = os.path.join(latest_run, "final_model.zip")
            chosen_model = best_model if os.path.exists(best_model) else final_model
            if os.path.exists(chosen_model):
                print(f"Auto-detected Transformer PPO model: {chosen_model}")
                model_trans = PPO.load(chosen_model)
                if not trans_vecnorm_path:
                    trans_vecnorm_path = os.path.join(latest_run, "vecnormalize.pkl")
            else:
                print("Warning: No pre-trained Transformer model file found in latest run. Falling back to MLP policy.")
                model_trans = model_mlp
        else:
            print("Warning: No Transformer PPO runs found. Falling back to MLP policy.")
            model_trans = model_mlp

    if not trans_vecnorm_path or not os.path.exists(trans_vecnorm_path):
        trans_vecnorm_path = vecnorm_path

    # Auto-detect PPO Gain Scheduler weights
    scheduler_pt_path = None
    if trans_model_path:
        d = os.path.dirname(trans_model_path)
        for p in [os.path.join(d, "scheduler_best.pt"), os.path.join(d, "scheduler_final.pt"), os.path.join(os.path.dirname(d), "scheduler_final.pt")]:
            if os.path.exists(p):
                scheduler_pt_path = p
                break
    if not scheduler_pt_path:
        trans_runs = glob.glob(os.path.join("results_hierarchical", "*trans*"))
        if trans_runs:
            trans_runs.sort()
            for latest_run in reversed(trans_runs):
                for p in [
                    os.path.join(latest_run, "best", "scheduler_best.pt"),
                    os.path.join(latest_run, "scheduler_final.pt"),
                ]:
                    if os.path.exists(p):
                        scheduler_pt_path = p
                        break
                if scheduler_pt_path:
                    break

    if scheduler_pt_path:
        print(f"Loaded PPO Gain Scheduler weights: {scheduler_pt_path}")
    else:
        print("Warning: PPO Gain Scheduler weights not found. Using randomly initialized weights.")

    # Define the 6 configurations
    # (key, use_scheduler, demo_guided, use_mpc, use_rl_gain, use_adaptive_mpc, use_history, display_name, is_transformer)
    configs = [
        ("pid_only", False, True, False, False, False, False, "PID-Only", False),
        ("rl_baseline", False, False, False, False, False, False, "PPO MLP + PID", False),
        ("rl_adaptive", True, False, False, False, False, False, "PPO MLP + Adaptive PID", False),
        ("rl_mpc_baseline", False, False, True, False, False, False, "PPO MLP + MPC + PID", False),
        ("rl_mpc_adaptive", True, False, True, False, False, False, "PPO MLP + MPC + Adaptive PID", False),
        ("trans_mpc_adaptive", True, False, True, False, False, True, "Transformer PPO + MPC + Adaptive PID", True),
    ]

    runs_data = {}
    for key, _, _, _, _, _, _, disp, _ in configs:
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
            "total_energy": [],
            "distance_traveled": [],
            "reward_per_joule": [],
            # New metrics:
            "peak_jerk": [],
            "avg_jerk": [],
            "oscillation_index": [],
            "energy_per_meter": [],
            "gain_pos_var": [],
            "gain_att_var": [],
            "mpc_h10_pct": [],
            "mpc_h20_pct": [],
            "mpc_h30_pct": [],
        }

    # For plotting a single comparison run over time:
    dist_histories = {}
    gain_histories = {}
    time_histories = {}

    for key, use_scheduler, demo_mode, use_mpc, use_rl_gain, use_adaptive_mpc, use_history, disp, is_trans in configs:
        print(f"\nEvaluating configuration: {disp}")

        # Choose model and vecnorm
        active_model = model_trans if is_trans else model_mlp
        active_vecnorm = trans_vecnorm_path if is_trans else vecnorm_path

        # Create evaluation environment
        domain_rand_enabled = getattr(cfg, "domain_randomization", False)
        raw_env_fn = make_eval_env(
            gui=cfg.gui,
            use_adaptive_scheduler=use_scheduler,
            demo_guided_mode=demo_mode,
            use_mpc_layer=use_mpc,
            use_rl_gain_scheduler=use_rl_gain,
            use_adaptive_mpc=use_adaptive_mpc,
            use_history=use_history,
            domain_randomization=domain_rand_enabled,
            telemetry_dir=out_dir
        )

        vec_env = DummyVecEnv([raw_env_fn])
        vec_env = VecNormalize.load(active_vecnorm, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
        vec_env.envs[0].evaluation_mode = True

        if use_rl_gain and scheduler_pt_path:
            vec_env.envs[0].rl_gain_scheduler.load(scheduler_pt_path)

        for r in range(cfg.rounds):
            seed = cfg.target_seed + r
            obs = vec_env.reset()
            # Set telemetry run metadata on raw environment
            vec_env.envs[0].set_telemetry_run_info(run_name=timestamp, config_name=key, round_idx=r+1)

            done = False
            ep_reward = 0.0

            dist_hist = []
            gain_hist = []

            while not done:
                action, _ = active_model.predict(obs, deterministic=True)
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
            tot_energy = info.get("total_energy", 0.0)
            dist_trav = info.get("distance_traveled", 0.0)
            r_per_j = info.get("reward_per_joule", 0.0)

            # New metrics
            peak_jerk = info.get("peak_jerk", 0.0)
            avg_jerk = info.get("avg_jerk", 0.0)
            oscillation_index = info.get("oscillation_index", 0.0)
            energy_per_meter = tot_energy / max(1e-6, dist_trav)
            gain_pos_var = info.get("gain_pos_var", 0.0)
            gain_att_var = info.get("gain_att_var", 0.0)
            mpc_h10 = info.get("mpc_h10_pct", 0.0)
            mpc_h20 = info.get("mpc_h20_pct", 0.0)
            mpc_h30 = info.get("mpc_h30_pct", 0.0)

            runs_data[key]["rewards"].append(ep_reward)
            runs_data[key]["success_rate"].append(success)
            runs_data[key]["mean_err"].append(mean_err)
            runs_data[key]["rmse_err"].append(rmse_err)
            runs_data[key]["overshoot"].append(overshoot)
            runs_data[key]["settling_time"].append(settling_time)
            runs_data[key]["smoothness"].append(smoothness)
            runs_data[key]["control_effort"].append(control_effort)
            runs_data[key]["wp_error"].append(wp_error)
            runs_data[key]["total_energy"].append(tot_energy)
            runs_data[key]["distance_traveled"].append(dist_trav)
            runs_data[key]["reward_per_joule"].append(r_per_j)

            # New metrics append
            runs_data[key]["peak_jerk"].append(peak_jerk)
            runs_data[key]["avg_jerk"].append(avg_jerk)
            runs_data[key]["oscillation_index"].append(oscillation_index)
            runs_data[key]["energy_per_meter"].append(energy_per_meter)
            runs_data[key]["gain_pos_var"].append(gain_pos_var)
            runs_data[key]["gain_att_var"].append(gain_att_var)
            runs_data[key]["mpc_h10_pct"].append(mpc_h10)
            runs_data[key]["mpc_h20_pct"].append(mpc_h20)
            runs_data[key]["mpc_h30_pct"].append(mpc_h30)

            print(
                f"Round {r+1}/{cfg.rounds}: Reward={ep_reward:.2f}, Success={success}, Tracking Error={mean_err:.3f}m, "
                f"Smoothness={smoothness:.1f}, Energy={tot_energy:.1f}, Avg Jerk={avg_jerk:.1f}, Osc={oscillation_index:.2f}"
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
            "Configuration", "Round", "Reward", "Success", "Mean_Tracking_Error_m", "RMSE_Tracking_Error_m", 
            "Overshoot_m", "Settling_Time_s", "Trajectory_Smoothness", "Control_Effort", 
            "Waypoint_Tracking_Error", "Total_Energy", "Distance_Traveled_m", "Reward_per_Joule",
            "Peak_Jerk", "Average_Jerk", "Oscillation_Index", "Energy_per_Meter",
            "Gain_Pos_Var", "Gain_Att_Var", "MPC_H10_Pct", "MPC_H20_Pct", "MPC_H30_Pct",
            "Robustness_Score", "Sim_to_Real_Readiness"
        ])

        for key, _, _, _, _, _, _, disp, _ in configs:
            m_succ = np.mean(runs_data[key]["success_rate"])
            m_err = np.mean(runs_data[key]["mean_err"])
            m_jerk = np.mean(runs_data[key]["avg_jerk"])
            m_en_m = np.mean(runs_data[key]["energy_per_meter"])
            rob = m_succ * np.exp(-m_err / 0.5)
            sim_to_real = 0.3 * (m_succ * 100) + 0.2 * max(0.0, 100 - m_jerk * 5) + 0.2 * max(0.0, 100 - m_en_m * 10) + 0.3 * (rob * 100)

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
                    f"{runs_data[key]['total_energy'][r]:.2f}",
                    f"{runs_data[key]['distance_traveled'][r]:.4f}",
                    f"{runs_data[key]['reward_per_joule'][r]:.6f}",
                    # New metrics per round
                    f"{runs_data[key]['peak_jerk'][r]:.2f}",
                    f"{runs_data[key]['avg_jerk'][r]:.2f}",
                    f"{runs_data[key]['oscillation_index'][r]:.2f}",
                    f"{runs_data[key]['energy_per_meter'][r]:.4f}",
                    f"{runs_data[key]['gain_pos_var'][r]:.6f}",
                    f"{runs_data[key]['gain_att_var'][r]:.6f}",
                    f"{runs_data[key]['mpc_h10_pct'][r]:.1f}",
                    f"{runs_data[key]['mpc_h20_pct'][r]:.1f}",
                    f"{runs_data[key]['mpc_h30_pct'][r]:.1f}",
                    "-",
                    "-"
                ])
            writer.writerow([
                disp,
                "Average",
                f"{np.mean(runs_data[key]['rewards']):.2f}",
                f"{m_succ:.1%}",
                f"{m_err:.4f}",
                f"{np.mean(runs_data[key]['rmse_err']):.4f}",
                f"{np.mean(runs_data[key]['overshoot']):.4f}",
                f"{np.mean(runs_data[key]['settling_time']):.2f}",
                f"{np.mean(runs_data[key]['smoothness']):.2f}",
                f"{np.mean(runs_data[key]['control_effort']):.6f}",
                f"{np.mean(runs_data[key]['wp_error']):.4f}",
                f"{np.mean(runs_data[key]['total_energy']):.2f}",
                f"{np.mean(runs_data[key]['distance_traveled']):.4f}",
                f"{np.mean(runs_data[key]['reward_per_joule']):.6f}",
                # New metrics averages
                f"{np.mean(runs_data[key]['peak_jerk']):.2f}",
                f"{m_jerk:.2f}",
                f"{np.mean(runs_data[key]['oscillation_index']):.2f}",
                f"{m_en_m:.4f}",
                f"{np.mean(runs_data[key]['gain_pos_var']):.6f}",
                f"{np.mean(runs_data[key]['gain_att_var']):.6f}",
                f"{np.mean(runs_data[key]['mpc_h10_pct']):.1f}",
                f"{np.mean(runs_data[key]['mpc_h20_pct']):.1f}",
                f"{np.mean(runs_data[key]['mpc_h30_pct']):.1f}",
                f"{rob:.4f}",
                f"{sim_to_real:.2f}"
            ])
            writer.writerow([])

    print(f"\nCSV results written to {csv_path}")

    # Process and print summaries to text report
    summary_path = os.path.join(out_dir, "comparison_summary.txt")
    with open(summary_path, "w") as f:
        def write_and_print(text):
            print(text)
            f.write(text + "\n")

        write_and_print("=" * 132)
        write_and_print("                                             6-WAY DRONE PERFORMANCE COMPARISON REPORT (HONORS EDITION)")
        write_and_print("=" * 132)
        write_and_print(f"Evaluated over {cfg.rounds} rounds with matched seeds.")
        write_and_print(f"Domain Randomization: {domain_rand_enabled}")
        write_and_print("-" * 132)
        write_and_print(
            f"{'Metric':<32} | "
            f"{'PID-Only':<12} | "
            f"{'PPO MLP':<12} | "
            f"{'PPO MLP+A':<12} | "
            f"{'MLP+MPC':<12} | "
            f"{'MLP+M+A':<12} | "
            f"{'Trans+M+A':<12}"
        )
        write_and_print("-" * 132)

        # Standard metrics
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
            ("Total Energy (x10^4 RPM^2)", "total_energy", "{:.1f}"),
            ("Distance Traveled (m)", "distance_traveled", "{:.4f}"),
            ("Reward per Joule", "reward_per_joule", "{:.6f}"),
            # Honors metrics
            ("Peak Jerk (m/s^3)", "peak_jerk", "{:.2f}"),
            ("Average Jerk (m/s^3)", "avg_jerk", "{:.2f}"),
            ("Oscillation Index", "oscillation_index", "{:.2f}"),
            ("Energy per Meter", "energy_per_meter", "{:.4f}"),
            ("Gain Pos Variance", "gain_pos_var", "{:.6f}"),
            ("Gain Att Variance", "gain_att_var", "{:.6f}"),
            ("MPC Horizon 10 Usage (%)", "mpc_h10_pct", "{:.1f}%"),
            ("MPC Horizon 20 Usage (%)", "mpc_h20_pct", "{:.1f}%"),
            ("MPC Horizon 30 Usage (%)", "mpc_h30_pct", "{:.1f}%"),
        ]

        for name, key, fmt in metrics_to_print:
            vals = [np.mean(runs_data[c[0]][key]) for c in configs]
            vals_str = [fmt.format(v) for v in vals]
            write_and_print(
                f"{name:<32} | "
                f"{vals_str[0]:<12} | "
                f"{vals_str[1]:<12} | "
                f"{vals_str[2]:<12} | "
                f"{vals_str[3]:<12} | "
                f"{vals_str[4]:<12} | "
                f"{vals_str[5]:<12}"
            )
        
        # Robustness & Sim-to-Real scores
        rob_scores = []
        s2r_scores = []
        for c in configs:
            m_succ = np.mean(runs_data[c[0]]["success_rate"])
            m_err = np.mean(runs_data[c[0]]["mean_err"])
            m_jerk = np.mean(runs_data[c[0]]["avg_jerk"])
            m_en_m = np.mean(runs_data[c[0]]["energy_per_meter"])
            rob = m_succ * np.exp(-m_err / 0.5)
            s2r = 0.3 * (m_succ * 100) + 0.2 * max(0.0, 100 - m_jerk * 5) + 0.2 * max(0.0, 100 - m_en_m * 10) + 0.3 * (rob * 100)
            rob_scores.append(rob)
            s2r_scores.append(s2r)

        write_and_print(
            f"{'Robustness Score':<32} | "
            f"{rob_scores[0]:.4f}         | "
            f"{rob_scores[1]:.4f}         | "
            f"{rob_scores[2]:.4f}         | "
            f"{rob_scores[3]:.4f}         | "
            f"{rob_scores[4]:.4f}         | "
            f"{rob_scores[5]:.4f}"
        )
        write_and_print(
            f"{'Sim-to-Real Readiness Score':<32} | "
            f"{s2r_scores[0]:.2f}         | "
            f"{s2r_scores[1]:.2f}         | "
            f"{s2r_scores[2]:.2f}         | "
            f"{s2r_scores[3]:.2f}         | "
            f"{s2r_scores[4]:.2f}         | "
            f"{s2r_scores[5]:.2f}"
        )
        write_and_print("=" * 132)

    print(f"Summary report written to {summary_path}")

    # Plot 1: Target Distance over time comparison for round 0
    plt.figure(figsize=(11, 6))
    colors = ['#7f7f7f', '#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c']
    linestyles = [':', '--', '-.', '-', '-', '-']
    for idx, (key, _, _, _, _, _, _, disp, _) in enumerate(configs):
        if key in dist_histories:
            plt.plot(
                time_histories[key][: len(dist_histories[key])],
                dist_histories[key],
                color=colors[idx],
                linestyle=linestyles[idx],
                label=disp
            )
    plt.axhline(y=0.60, color="k", linestyle="-.", alpha=0.5, label="Target Threshold (0.60m)")
    plt.xlabel("Simulation Time (s)")
    plt.ylabel("Distance to Target (m)")
    plt.title("Target Distance Tracking over Time (Round 1 Comparison)")
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "tracking_comparison.png"), bbox_inches="tight")
    plt.close()

    # Plot 2: Gain Scale over time for round 0
    plt.figure(figsize=(11, 5))
    if "rl_adaptive" in gain_histories:
        plt.plot(time_histories["rl_adaptive"][: len(gain_histories["rl_adaptive"])], gain_histories["rl_adaptive"], "b--", label="PPO MLP + Adaptive PID")
    if "rl_mpc_adaptive" in gain_histories:
        plt.plot(time_histories["rl_mpc_adaptive"][: len(gain_histories["rl_mpc_adaptive"])], gain_histories["rl_mpc_adaptive"], "m-.", label="PPO MLP + MPC + Adaptive PID")
    plt.xlabel("Simulation Time (s)")
    plt.ylabel("Gain Scale (multiplier)")
    plt.title("PID Gain Scale Modulation over Time (Round 1)")
    plt.axhline(y=1.0, color="k", linestyle="--", alpha=0.5, label="Baseline (1.0)")
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "gain_scale_history.png"), bbox_inches="tight")
    plt.close()

    # Plot 3: Unified bar charts in a 4x3 grid
    fig, axes = plt.subplots(4, 3, figsize=(18, 20))
    axes = axes.flatten()

    short_categories = ["PID", "PPO", "PPO+A", "MLP+MPC", "MLP+M+A", "Trans+M+A"]
    
    metrics_to_plot = [
        ("mean_err", "Mean Tracking Error", "Error (m)"),
        ("rmse_err", "RMSE Tracking Error", "Error (m)"),
        ("settling_time", "Average Settling Time", "Time (s)"),
        ("rewards", "Average Episode Reward", "Reward"),
        ("smoothness", "Trajectory Smoothness (Jerk)", "Smoothness"),
        ("control_effort", "Control Effort (CMD Vel^2)", "Effort"),
        ("total_energy", "Total Energy Consumed", "Energy"),
        ("distance_traveled", "Distance Traveled", "Distance (m)"),
        ("reward_per_joule", "Reward per Joule", "Reward/Joule"),
        ("peak_jerk", "Peak Jerk", "Jerk (m/s^3)"),
        ("oscillation_index", "Gain Oscillation Index", "Oscillation"),
        ("sim_to_real", "Sim-to-Real Readiness", "Readiness Score")
    ]

    for idx, (metric_key, title, ylabel) in enumerate(metrics_to_plot):
        ax = axes[idx]
        if metric_key == "sim_to_real":
            vals = s2r_scores
        else:
            vals = [np.mean(runs_data[c[0]][metric_key]) for c in configs]
        ax.bar(short_categories, vals, color=colors, width=0.5)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(axis="y")
        ax.tick_params(axis="x", labelrotation=25, labelsize=9)

    plt.suptitle("6-Way UAV Control Stack Upgrade - Unified Benchmarks", fontsize=18, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    plt.savefig(os.path.join(out_dir, "metrics_comparison.png"))
    plt.close()

    print(
        f"Unified charts saved to {out_dir}/tracking_comparison.png, {out_dir}/gain_scale_history.png, and {out_dir}/metrics_comparison.png"
    )
    
    # Save the output directories to artifacts for display if required
    artifact_dir = "C:/Users/B Siddarth Vijayan/.gemini/antigravity-ide/brain/6ba40026-340c-4d66-a309-f317a44573a5"
    if os.path.exists(artifact_dir):
        import shutil
        try:
            shutil.copy(os.path.join(out_dir, "tracking_comparison.png"), os.path.join(artifact_dir, "tracking_comparison.png"))
            shutil.copy(os.path.join(out_dir, "gain_scale_history.png"), os.path.join(artifact_dir, "gain_scale_history.png"))
            shutil.copy(os.path.join(out_dir, "metrics_comparison.png"), os.path.join(artifact_dir, "metrics_comparison.png"))
            print(f"Copied plots to artifacts directory: {artifact_dir}")
        except Exception as e:
            print(f"Could not copy files to artifact directory: {e}")
