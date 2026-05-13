import os
import time

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
    vec_env = DummyVecEnv([make_eval_env(gui=gui)])
    vec_env = VecNormalize.load(vecnorm_path, vec_env)
    vec_env.training = False
    vec_env.norm_reward = False

    model = PPO.load(model_path)

    for ep in range(episodes):
        obs = vec_env.reset()
        done = False
        ep_reward = 0.0
        hit_target = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = vec_env.step(action)
            ep_reward += float(reward[0])
            done = bool(dones[0])
            if infos and isinstance(infos[0], dict):
                hit_target = hit_target or bool(infos[0].get("success", 0.0) > 0.5)
            if gui:
                time.sleep(0.035)
        status = "HIT" if hit_target else "MISS"
        print(f"Episode {ep+1}/{episodes} reward={ep_reward:.2f} target={status}")

    vec_env.close()


if __name__ == "__main__":
    base = os.path.join("results_hierarchical")
    print("Call evaluate(model_path, vecnorm_path) from code or set paths manually.")
