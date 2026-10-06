import os
import argparse
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from hierarchical_drone.config.settings import ActionConfig, SensorConfig, SimConfig, TaskConfig
from hierarchical_drone.env.hierarchical_nav_env import HierarchicalNavEnv
from hierarchical_drone.rl.transformer_feature_extractor import TransformerFeatureExtractor


class SchedulerCallback(BaseCallback):
    def __init__(self, out_dir: str, save_freq: int = 20_000, verbose: int = 0):
        super().__init__(verbose)
        self.out_dir = out_dir
        self.save_freq = save_freq

    def _on_step(self) -> bool:
        # Checkpoint save
        if self.n_calls % self.save_freq == 0:
            envs = self.training_env.envs
            for idx, env in enumerate(envs):
                real_env = env.unwrapped
                if hasattr(real_env, "rl_gain_scheduler") and real_env.rl_gain_scheduler is not None:
                    checkpoint_dir = os.path.join(self.out_dir, "checkpoints")
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    path = os.path.join(checkpoint_dir, f"scheduler_checkpoint_{self.n_calls}.pt")
                    real_env.rl_gain_scheduler.save(path)
                    if self.verbose > 0:
                        print(f"[INFO] Saved scheduler checkpoint to {path}")

        # Sync best scheduler state
        best_model_path = os.path.join(self.out_dir, "best", "best_model.zip")
        if os.path.exists(best_model_path):
            envs = self.training_env.envs
            for env in envs:
                real_env = env.unwrapped
                if hasattr(real_env, "rl_gain_scheduler") and real_env.rl_gain_scheduler is not None:
                    best_dir = os.path.join(self.out_dir, "best")
                    os.makedirs(best_dir, exist_ok=True)
                    path = os.path.join(best_dir, "scheduler_best.pt")
                    real_env.rl_gain_scheduler.save(path)
        return True


def make_env(
    gui: bool = False,
    use_history: bool = False,
    use_rl_gain_scheduler: bool = False,
    use_mpc_layer: bool = False,
    use_adaptive_mpc: bool = False,
    domain_randomization: bool = False
):
    def _thunk():
        env = HierarchicalNavEnv(
            sim=SimConfig(gui=gui),
            task=TaskConfig(),
            sensor_cfg=SensorConfig(),
            action_cfg=ActionConfig(),
            use_history=use_history,
            use_rl_gain_scheduler=use_rl_gain_scheduler,
            use_mpc_layer=use_mpc_layer,
            use_adaptive_mpc=use_adaptive_mpc,
            domain_randomization=domain_randomization,
        )
        return Monitor(env)

    return _thunk


def train(
    total_timesteps: int = 500_000,
    run_name: str = None,
    policy_type: str = "mlp",
    use_rl_gain_scheduler: bool = False,
    use_mpc_layer: bool = False,
    use_adaptive_mpc: bool = False,
    domain_randomization: bool = True
):
    run_name = run_name or datetime.now().strftime(f"run_{policy_type}_%Y%m%d_%H%M%S")
    out_dir = os.path.join("results_hierarchical", run_name)
    os.makedirs(out_dir, exist_ok=True)

    use_history = (policy_type == "transformer")

    vec_env = DummyVecEnv([
        make_env(
            gui=False,
            use_history=use_history,
            use_rl_gain_scheduler=use_rl_gain_scheduler,
            use_mpc_layer=use_mpc_layer,
            use_adaptive_mpc=use_adaptive_mpc,
            domain_randomization=domain_randomization
        )
    ])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # For evaluation, we disable domain randomization for consistency in tracking metrics
    eval_env = DummyVecEnv([
        make_env(
            gui=False,
            use_history=use_history,
            use_rl_gain_scheduler=use_rl_gain_scheduler,
            use_mpc_layer=use_mpc_layer,
            use_adaptive_mpc=use_adaptive_mpc,
            domain_randomization=False
        )
    ])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0, training=False)

    policy_kwargs = {}
    if policy_type == "transformer":
        policy_kwargs = {
            "features_extractor_class": TransformerFeatureExtractor,
            "features_extractor_kwargs": {"features_dim": 128},
        }

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        verbose=1,
        n_steps=2048,
        batch_size=256,
        gamma=0.995,
        gae_lambda=0.95,
        learning_rate=3e-4,
        policy_kwargs=policy_kwargs,
        tensorboard_log=os.path.join(out_dir, "tb"),
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=20_000,
        save_path=os.path.join(out_dir, "checkpoints"),
        name_prefix=f"ppo_{policy_type}"
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(out_dir, "best"),
        log_path=os.path.join(out_dir, "eval"),
        eval_freq=10_000,
        deterministic=True,
    )
    
    callbacks = [checkpoint_cb, eval_cb]
    if use_rl_gain_scheduler:
        scheduler_cb = SchedulerCallback(out_dir=out_dir, save_freq=20_000, verbose=1)
        callbacks.append(scheduler_cb)

    model.learn(total_timesteps=total_timesteps, callback=callbacks)

    model.save(os.path.join(out_dir, "final_model"))
    if use_rl_gain_scheduler:
        envs = vec_env.envs
        for idx, env in enumerate(envs):
            real_env = env.unwrapped
            if hasattr(real_env, "rl_gain_scheduler") and real_env.rl_gain_scheduler is not None:
                path = os.path.join(out_dir, "scheduler_final.pt")
                real_env.rl_gain_scheduler.save(path)
                print(f"[INFO] Saved final scheduler to {path}")

    vec_env.save(os.path.join(out_dir, "vecnormalize.pkl"))
    vec_env.close()
    eval_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO policy (MLP or Transformer)")
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--policy_type", type=str, choices=["mlp", "transformer"], default="mlp")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--use_rl_gain_scheduler", action="store_true", help="Enable PPO PID gain scheduler")
    parser.add_argument("--use_mpc_layer", action="store_true", help="Enable MPC path tracking layer")
    parser.add_argument("--use_adaptive_mpc", action="store_true", help="Enable adaptive horizon switching in MPC")
    parser.add_argument("--no_domain_randomization", action="store_true", help="Disable domain randomization")
    args = parser.parse_args()
    
    train(
        total_timesteps=args.timesteps,
        run_name=args.run_name,
        policy_type=args.policy_type,
        use_rl_gain_scheduler=args.use_rl_gain_scheduler,
        use_mpc_layer=args.use_mpc_layer,
        use_adaptive_mpc=args.use_adaptive_mpc,
        domain_randomization=not args.no_domain_randomization
    )
