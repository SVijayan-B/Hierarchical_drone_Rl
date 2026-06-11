import os
import time
import csv
import numpy as np
import matplotlib

matplotlib.use("Agg")  # Safe headless execution
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from hierarchical_drone.config.settings import ActionConfig, SensorConfig, SimConfig, TaskConfig
from hierarchical_drone.env.hierarchical_nav_env import HierarchicalNavEnv


def make_eval_env(gui=True):
    def _thunk():
        return HierarchicalNavEnv(
            sim=SimConfig(gui=gui),
            task=TaskConfig(),
            sensor_cfg=SensorConfig(),
            action_cfg=ActionConfig(),
        )

    return _thunk


def evaluate(model_path, vecnorm_path, episodes=5, gui=True):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("results", f"metrics_plots_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    vec_env = DummyVecEnv([make_eval_env(gui=gui)])
    vec_env = VecNormalize.load(vecnorm_path, vec_env)
    vec_env.training = False
    vec_env.norm_reward = False

    model = PPO.load(model_path)

    summary_data = []

    for ep in range(episodes):
        obs = vec_env.reset()
        done = False
        ep_reward = 0.0
        hit_target = False

        steps_log = []
        step_idx = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = vec_env.step(action)
            ep_reward += float(reward[0])
            done = bool(dones[0])

            info = infos[0]
            if infos and isinstance(info, dict):
                hit_target = hit_target or bool(info.get("success", 0.0) > 0.5)

                vel = info.get("vel", [0.0, 0.0, 0.0])
                rpy = info.get("rpy", [0.0, 0.0, 0.0])
                rates = info.get("rates", [0.0, 0.0, 0.0])
                rpm = info.get("rpm", [0.0, 0.0, 0.0, 0.0])

                vel_xy = float(np.linalg.norm(vel[0:2]))
                vel_z = float(abs(vel[2]))
                tilt_abs = float(np.rad2deg(np.sqrt(rpy[0] ** 2 + rpy[1] ** 2)))
                ang_rate_norm = float(np.rad2deg(np.linalg.norm(rates)))
                power_proxy = float(np.mean(rpm) / 10000.0)

                steps_log.append((step_idx, vel_xy, vel_z, tilt_abs, ang_rate_norm, power_proxy))
                step_idx += 1

            if gui:
                time.sleep(0.035)

        info = infos[0]
        status = "HIT" if hit_target else "MISS"
        mean_err = info.get("mean_tracking_error", 0.0)
        overshoot = info.get("overshoot", 0.0)
        settling = info.get("settling_time", 0.0)
        rmse_err = info.get("rmse_tracking_error", 0.0)

        print(
            f"Episode {ep+1}/{episodes} reward={ep_reward:.2f} target={status} tracking_error={mean_err:.3f}m overshoot={overshoot:.3f}m settling_time={settling:.2f}s"
        )

        summary_data.append((ep + 1, ep_reward, status, mean_err, rmse_err, overshoot, settling))

        # Save detailed step-by-step metrics
        csv_filepath = os.path.join(out_dir, f"episode_{ep+1}_metrics.csv")
        with open(csv_filepath, "w", newline="") as csv_f:
            writer = csv.writer(csv_f)
            writer.writerow(["step", "vel_xy", "vel_z", "tilt_abs", "ang_rate_norm", "power_proxy"])
            writer.writerows(steps_log)

        # Unpack step data for plotting
        if steps_log:
            steps = [row[0] for row in steps_log]
            vels_xy = [row[1] for row in steps_log]
            vels_z = [row[2] for row in steps_log]
            tilts = [row[3] for row in steps_log]
            ang_rates = [row[4] for row in steps_log]
            power_proxies = [row[5] for row in steps_log]

            # Plot Power Proxy
            plt.figure()
            plt.plot(steps, power_proxies, "g-", label="Power Proxy")
            plt.xlabel("Step")
            plt.ylabel("Power Proxy")
            plt.title(f"Episode {ep+1} Power Proxy")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"episode_{ep+1}_power.png"))
            plt.close()

            # Plot Stability (Tilt & Angular Rate)
            fig, ax1 = plt.subplots()
            ax1.plot(steps, tilts, "b-", label="Tilt (deg)")
            ax1.set_xlabel("Step")
            ax1.set_ylabel("Tilt (degrees)", color="b")
            ax1.tick_params(axis="y", labelcolor="b")

            ax2 = ax1.twinx()
            ax2.plot(steps, ang_rates, "r-", label="Angular Rate (deg/s)")
            ax2.set_ylabel("Angular Rate (degrees/s)", color="r")
            ax2.tick_params(axis="y", labelcolor="r")

            plt.title(f"Episode {ep+1} Stability (Tilt & Angular Rate)")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"episode_{ep+1}_stability.png"))
            plt.close()

            # Plot Velocity
            plt.figure()
            plt.plot(steps, vels_xy, "r-", label="Vel XY (m/s)")
            plt.plot(steps, vels_z, "b--", label="Vel Z (m/s)")
            plt.xlabel("Step")
            plt.ylabel("Velocity (m/s)")
            plt.title(f"Episode {ep+1} Velocities")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"episode_{ep+1}_velocity.png"))
            plt.close()

    # Save summary CSV
    summary_csv_filepath = os.path.join(out_dir, "episode_summary.csv")
    with open(summary_csv_filepath, "w", newline="") as csv_f:
        writer = csv.writer(csv_f)
        writer.writerow([
            "Episode",
            "Reward",
            "Success",
            "Mean_Tracking_Error_m",
            "RMSE_Tracking_Error_m",
            "Overshoot_m",
            "Settling_Time_s",
        ])
        writer.writerows(summary_data)

    vec_env.close()
    print(f"\nAll simulation results saved under: {out_dir}")


if __name__ == "__main__":
    base = os.path.join("results_hierarchical")
    print("Call evaluate(model_path, vecnorm_path) from code or set paths manually.")
