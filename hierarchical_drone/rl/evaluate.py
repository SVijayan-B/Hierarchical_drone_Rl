import os
import time
import json
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from hierarchical_drone.config.settings import ActionConfig, SensorConfig, SimConfig, TaskConfig
from hierarchical_drone.config.settings import (
    VELOCITY_XY_GAINS,
    VELOCITY_Z_GAINS,
    ATT_ROLL_GAINS,
    ATT_PITCH_GAINS,
    ATT_YAW_GAINS,
    RATE_ROLL_GAINS,
    RATE_PITCH_GAINS,
    RATE_YAW_GAINS,
)
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
    vec_env = DummyVecEnv([make_eval_env(gui=gui)])
    vec_env = VecNormalize.load(vecnorm_path, vec_env)
    vec_env.training = False
    vec_env.norm_reward = False

    model = PPO.load(model_path)

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    results_root = os.path.join(project_root, "results")
    os.makedirs(results_root, exist_ok=True)
    metrics_dir = os.path.join(results_root, "metrics_plots_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(metrics_dir, exist_ok=True)
    episode_rewards = []
    episode_stability = []

    for ep in range(episodes):
        obs = vec_env.reset()
        done = False
        ep_reward = 0.0
        hit_target = False
        vxy_series = []
        vz_series = []
        tilt_series = []
        ang_rate_series = []
        power_series = []
        t_series = []
        step_idx = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = vec_env.step(action)
            ep_reward += float(reward[0])
            done = bool(dones[0])
            if infos and isinstance(infos[0], dict):
                hit_target = hit_target or bool(infos[0].get("success", 0.0) > 0.5)
                vxy_series.append(float(infos[0].get("vel_xy", 0.0)))
                vz_series.append(float(infos[0].get("vel_z", 0.0)))
                tilt_series.append(float(infos[0].get("tilt_abs", 0.0)))
                ang_rate_series.append(float(infos[0].get("ang_rate_norm", 0.0)))
                power_series.append(float(infos[0].get("power_proxy", 0.0)))
                t_series.append(step_idx)
                step_idx += 1
            if gui:
                time.sleep(0.035)
        status = "HIT" if hit_target else "MISS"
        print(f"Episode {ep+1}/{episodes} reward={ep_reward:.2f} target={status}")
        episode_rewards.append(float(ep_reward))

        if len(t_series) > 2:
            # Velocity plot
            plt.figure(figsize=(9, 4))
            plt.plot(t_series, vxy_series, label="Horizontal speed proxy (m/s)")
            plt.plot(t_series, vz_series, label="Vertical speed proxy (m/s)")
            plt.title(f"Episode {ep+1}: Velocity")
            plt.xlabel("Step")
            plt.ylabel("Speed")
            plt.grid(alpha=0.35)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(metrics_dir, f"episode_{ep+1}_velocity.png"), dpi=150)
            plt.close()

            # Stability plot (proxy)
            plt.figure(figsize=(9, 4))
            plt.plot(t_series, tilt_series, label="Tilt stability proxy")
            plt.plot(t_series, ang_rate_series, label="Angular-rate stability proxy")
            plt.title(f"Episode {ep+1}: Stability Proxies")
            plt.xlabel("Step")
            plt.ylabel("Proxy value")
            plt.grid(alpha=0.35)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(metrics_dir, f"episode_{ep+1}_stability.png"), dpi=150)
            plt.close()

            # Power plot
            plt.figure(figsize=(9, 4))
            plt.plot(t_series, power_series, label="Power proxy (|action|)")
            plt.title(f"Episode {ep+1}: Power Consumption Proxy")
            plt.xlabel("Step")
            plt.ylabel("Proxy value")
            plt.grid(alpha=0.35)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(metrics_dir, f"episode_{ep+1}_power.png"), dpi=150)
            plt.close()

            csv_path = os.path.join(metrics_dir, f"episode_{ep+1}_metrics.csv")
            with open(csv_path, "w", encoding="utf-8") as f:
                f.write("step,vel_xy,vel_z,tilt_abs,ang_rate_norm,power_proxy\n")
                for i in range(len(t_series)):
                    f.write(
                        f"{t_series[i]},{vxy_series[i]},{vz_series[i]},"
                        f"{tilt_series[i]},{ang_rate_series[i]},{power_series[i]}\n"
                    )
            episode_stability.append(
                {
                    "episode": ep + 1,
                    "target_hit": status,
                    "reward": float(ep_reward),
                    "vel_xy_mean": float(np.mean(vxy_series)),
                    "vel_xy_std": float(np.std(vxy_series)),
                    "vel_z_mean": float(np.mean(vz_series)),
                    "tilt_abs_mean": float(np.mean(tilt_series)),
                    "tilt_abs_max": float(np.max(tilt_series)),
                    "ang_rate_mean": float(np.mean(ang_rate_series)),
                    "power_proxy_mean": float(np.mean(power_series)),
                    "power_proxy_max": float(np.max(power_series)),
                }
            )

    # Overall reward graph across episodes.
    if len(episode_rewards) > 0:
        x = np.arange(1, len(episode_rewards) + 1)
        plt.figure(figsize=(8, 4))
        plt.plot(x, episode_rewards, marker="o", color="purple", label="Episode reward")
        plt.title("PPO Evaluation Reward per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.grid(alpha=0.35)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(metrics_dir, "reward_per_episode.png"), dpi=150)
        plt.close()

    # Save PPO details + gains + stability summary.
    ppo_info = {
        "model_path": os.path.abspath(model_path),
        "vecnorm_path": os.path.abspath(vecnorm_path),
        "episodes": int(episodes),
        "algorithm": "PPO (Stable-Baselines3)",
        "policy_class": "MlpPolicy",
        "note": "Detailed PPO hyperparameters are in training config used in train_ppo.py.",
    }
    gains_info = {
        "VELOCITY_XY_GAINS": VELOCITY_XY_GAINS.__dict__,
        "VELOCITY_Z_GAINS": VELOCITY_Z_GAINS.__dict__,
        "ATT_ROLL_GAINS": ATT_ROLL_GAINS.__dict__,
        "ATT_PITCH_GAINS": ATT_PITCH_GAINS.__dict__,
        "ATT_YAW_GAINS": ATT_YAW_GAINS.__dict__,
        "RATE_ROLL_GAINS": RATE_ROLL_GAINS.__dict__,
        "RATE_PITCH_GAINS": RATE_PITCH_GAINS.__dict__,
        "RATE_YAW_GAINS": RATE_YAW_GAINS.__dict__,
    }

    with open(os.path.join(metrics_dir, "ppo_details.json"), "w", encoding="utf-8") as f:
        json.dump(ppo_info, f, indent=2)
    with open(os.path.join(metrics_dir, "controller_gains.json"), "w", encoding="utf-8") as f:
        json.dump(gains_info, f, indent=2)
    with open(os.path.join(metrics_dir, "episode_stability_summary.json"), "w", encoding="utf-8") as f:
        json.dump(episode_stability, f, indent=2)

    print(f"[INFO] Saved metric plots to: {metrics_dir}")

    vec_env.close()


if __name__ == "__main__":
    base = os.path.join("results_hierarchical")
    print("Call evaluate(model_path, vecnorm_path) from code or set paths manually.")
