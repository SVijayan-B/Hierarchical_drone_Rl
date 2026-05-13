import os
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from hierarchical_drone.config.settings import ActionConfig, SensorConfig, SimConfig, TaskConfig
from hierarchical_drone.env.hierarchical_nav_env import HierarchicalNavEnv


def make_env(gui=False):
    def _thunk():
        env = HierarchicalNavEnv(
            sim=SimConfig(gui=gui),
            task=TaskConfig(),
            sensor_cfg=SensorConfig(),
            action_cfg=ActionConfig(),
        )
        return Monitor(env)

    return _thunk


def train(total_timesteps=500_000, run_name=None):
    run_name = run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    out_dir = os.path.join("results_hierarchical", run_name)
    os.makedirs(out_dir, exist_ok=True)

    vec_env = DummyVecEnv([make_env(gui=False)])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    eval_env = DummyVecEnv([make_env(gui=False)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0, training=False)

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        verbose=1,
        n_steps=2048,
        batch_size=256,
        gamma=0.995,
        gae_lambda=0.95,
        learning_rate=3e-4,
        tensorboard_log=os.path.join(out_dir, "tb"),
    )

    checkpoint_cb = CheckpointCallback(save_freq=20_000, save_path=os.path.join(out_dir, "checkpoints"), name_prefix="ppo_hier")
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(out_dir, "best"),
        log_path=os.path.join(out_dir, "eval"),
        eval_freq=10_000,
        deterministic=True,
    )

    model.learn(total_timesteps=total_timesteps, callback=[checkpoint_cb, eval_cb])

    model.save(os.path.join(out_dir, "final_model"))
    vec_env.save(os.path.join(out_dir, "vecnormalize.pkl"))
    vec_env.close()
    eval_env.close()


if __name__ == "__main__":
    train()
