"""RL+PID stack: PPO high-level commands + DSLPIDControl low-level stabilization."""

from __future__ import annotations

import time
from typing import List, Optional

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from hierarchical_drone.benchmark.metrics import CompareConfig, RoundMetrics, finalize_round_metrics
from hierarchical_drone.config.settings import ActionConfig, SensorConfig, SimConfig, TaskConfig
from hierarchical_drone.env.hierarchical_nav_env import HierarchicalNavEnv


def _make_env(gui: bool, episode_sec: int = 40) -> HierarchicalNavEnv:
    sim = SimConfig(gui=gui, episode_sec=episode_sec)
    env = HierarchicalNavEnv(
        sim=sim,
        task=TaskConfig(),
        sensor_cfg=SensorConfig(),
        action_cfg=ActionConfig(),
    )
    env.demo_guided_mode = False
    env.guidance_blend = 0.0
    return env


def _unwrap_hier_env(vec_env: VecNormalize) -> HierarchicalNavEnv:
    env = vec_env.venv.envs[0]
    if hasattr(env, "unwrapped"):
        env = env.unwrapped
    if not isinstance(env, HierarchicalNavEnv):
        raise TypeError(f"Expected HierarchicalNavEnv, got {type(env)}")
    return env


def run_rl_pid_rounds(
    targets_xy: List[List[float]],
    cfg: CompareConfig,
    model_path: str,
    vecnorm_path: str,
) -> List[RoundMetrics]:
    episode_sec = int(cfg.round_timeout_sec)

    def _thunk():
        return Monitor(_make_env(cfg.gui, episode_sec))

    vec_env = DummyVecEnv([_thunk])
    vec_env = VecNormalize.load(vecnorm_path, vec_env)
    vec_env.training = False
    vec_env.norm_reward = False
    model = PPO.load(model_path)

    hier_ref = _unwrap_hier_env(vec_env)
    max_steps = int(cfg.round_timeout_sec * hier_ref.sim_cfg.ctrl_freq)
    results: List[RoundMetrics] = []

    for r, tgt_xy in enumerate(targets_xy):
        target_xyz = np.array([tgt_xy[0], tgt_xy[1], cfg.target_height], dtype=np.float32)
        hier_env = _unwrap_hier_env(vec_env)
        hier_env.fixed_target_xyz = target_xyz.copy()
        obs = vec_env.reset()

        start_t = time.time()
        hit_time: Optional[float] = None
        settle_time: Optional[float] = None
        power_samples: List[float] = []
        done = False
        steps = 0

        while not done and steps < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, dones, infos = vec_env.step(action)
            steps += 1
            done = bool(dones[0])
            info = infos[0] if infos else {}

            power_samples.append(float(info.get("power_proxy_step", 0.0)))
            dist = float(info.get("distance", 999.0))
            alt_err = float(info.get("altitude_error", 999.0))

            if settle_time is None and dist <= cfg.stabilize_dist_m and alt_err <= cfg.stabilize_z_tol_m:
                settle_time = time.time() - start_t

            if hit_time is None and dist <= cfg.hit_radius_m:
                hit_time = time.time() - start_t

            if cfg.gui:
                time.sleep(0.01)

        results.append(
            finalize_round_metrics(r, hit_time, settle_time, power_samples, cfg.round_timeout_sec)
        )

    vec_env.close()
    return results
