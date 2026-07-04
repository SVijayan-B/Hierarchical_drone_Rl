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


def linear_schedule(initial_value: float):
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func


class CurriculumCallback(BaseCallback):
    def __init__(self, total_timesteps: int, verbose: int = 0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.current_phase = 1

    def _on_step(self) -> bool:
        steps = self.model.num_timesteps
        pct = steps / max(1, self.total_timesteps)
        
        # Linearly decay entropy coefficient from 0.01 to 0.0
        self.model.ent_coef = max(0.0, (1.0 - pct) * 0.01)

        # 5 curriculum phases split evenly
        if pct < 0.20:
            phase = 1
        elif pct < 0.40:
            phase = 2
        elif pct < 0.60:
            phase = 3
        elif pct < 0.80:
            phase = 4
        else:
            phase = 5
            
        if phase != self.current_phase:
            self.current_phase = phase
            print(f"\n[CurriculumCallback] Transitioned to Curriculum Phase {phase} ({steps}/{self.total_timesteps} steps, {pct*100:.1f}%)")
            
        for env in self.training_env.envs:
            env.unwrapped.curriculum_phase = phase
            
        return True


def make_env(
    gui: bool = False,
    use_history: bool = False,
    use_mpc_layer: bool = False,
    use_adaptive_mpc: bool = False,
    domain_randomization: bool = False,
    demo_guided_mode: bool = False,
    use_robust_obs: bool = True
):
    def _thunk():
        env = HierarchicalNavEnv(
            sim=SimConfig(gui=gui),
            task=TaskConfig(),
            sensor_cfg=SensorConfig(),
            action_cfg=ActionConfig(),
            use_history=use_history,
            use_rl_gain_scheduler=False,
            use_mpc_layer=use_mpc_layer,
            use_adaptive_mpc=use_adaptive_mpc,
            domain_randomization=domain_randomization,
            demo_guided_mode=demo_guided_mode,
            use_robust_obs=use_robust_obs
        )
        return Monitor(env)

    return _thunk


def train(
    total_timesteps: int = 1500_000,
    run_name: str = None,
    policy_type: str = "mlp",
    use_mpc_layer: bool = False,
    use_adaptive_mpc: bool = False,
    domain_randomization: bool = True,
    use_robust_obs: bool = True,
    gui: bool = False
):
    run_name = run_name or datetime.now().strftime(f"run_{policy_type}_%Y%m%d_%H%M%S")
    out_dir = os.path.join("results_hierarchical", run_name)
    os.makedirs(out_dir, exist_ok=True)

    use_history = (policy_type == "transformer")

    vec_env = DummyVecEnv([
        make_env(
            gui=gui,
            use_history=use_history,
            use_mpc_layer=use_mpc_layer,
            use_adaptive_mpc=use_adaptive_mpc,
            domain_randomization=domain_randomization,
            demo_guided_mode=False,
            use_robust_obs=use_robust_obs
        )
    ])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # For evaluation, we disable domain randomization for consistency in tracking metrics
    # But we still keep robust observations to match the policy input shape
    eval_env = DummyVecEnv([
        make_env(
            gui=False,
            use_history=use_history,
            use_mpc_layer=use_mpc_layer,
            use_adaptive_mpc=use_adaptive_mpc,
            domain_randomization=False,
            demo_guided_mode=False,
            use_robust_obs=use_robust_obs
        )
    ])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0, training=False)

    policy_kwargs = {}
    if policy_type == "transformer":
        policy_kwargs = {
            "features_extractor_class": TransformerFeatureExtractor,
            "features_extractor_kwargs": {"features_dim": 128},
        }

    # Linear decay schedule for learning rate
    lr_schedule = linear_schedule(3e-4)

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        verbose=1,
        n_steps=2048,
        batch_size=256,
        gamma=0.995,
        gae_lambda=0.95,
        learning_rate=lr_schedule,
        ent_coef=0.01,
        max_grad_norm=0.5,
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
    
    curriculum_cb = CurriculumCallback(total_timesteps=total_timesteps, verbose=1)
    callbacks = [checkpoint_cb, eval_cb, curriculum_cb]

    model.learn(total_timesteps=total_timesteps, callback=callbacks)

    model.save(os.path.join(out_dir, "final_model"))

    vec_env.save(os.path.join(out_dir, "vecnormalize.pkl"))
    vec_env.close()
    eval_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO policy (MLP or Transformer)")
    parser.add_argument("--timesteps", type=int, default=1500_000)
    parser.add_argument("--policy_type", type=str, choices=["mlp", "transformer"], default="mlp")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--use_mpc_layer", action="store_true", help="Enable MPC path tracking layer")
    parser.add_argument("--use_adaptive_mpc", action="store_true", help="Enable adaptive horizon switching in MPC")
    parser.add_argument("--no_domain_randomization", action="store_true", help="Disable domain randomization")
    parser.add_argument("--gui", action="store_true", help="Enable GUI visualization during training")
    args = parser.parse_args()
    
    train(
        total_timesteps=args.timesteps,
        run_name=args.run_name,
        policy_type=args.policy_type,
        use_mpc_layer=args.use_mpc_layer,
        use_adaptive_mpc=args.use_adaptive_mpc,
        domain_randomization=not args.no_domain_randomization,
        use_robust_obs=True,
        gui=args.gui
    )
