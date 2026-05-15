"""PID-only baseline: DSLPIDControl chases target directly (no RL layer)."""

from __future__ import annotations

import time
from typing import List, Optional

import numpy as np
import pybullet as p
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics

from hierarchical_drone.benchmark.metrics import CompareConfig, RoundMetrics, finalize_round_metrics
from hierarchical_drone.config.settings import SimConfig


def run_pid_rounds(targets_xy: List[List[float]], cfg: CompareConfig) -> List[RoundMetrics]:
    sim = SimConfig(gui=cfg.gui, episode_sec=int(cfg.round_timeout_sec))
    ctrl_freq = sim.ctrl_freq
    ctrl_dt = 1.0 / ctrl_freq
    max_steps = int(cfg.round_timeout_sec * ctrl_freq)
    takeoff_steps = int(4.0 * ctrl_freq)

    env = CtrlAviary(
        drone_model=DroneModel.CF2X,
        num_drones=1,
        initial_xyzs=np.array([[0.0, 0.0, 0.2]]),
        initial_rpys=np.array([[0.0, 0.0, 0.0]]),
        physics=Physics.PYB,
        pyb_freq=sim.pyb_freq,
        ctrl_freq=ctrl_freq,
        gui=cfg.gui,
        obstacles=False,
        user_debug_gui=False,
        record=False,
    )
    client = env.getPyBulletClient()
    controller = DSLPIDControl(drone_model=DroneModel.CF2X)
    action = np.zeros((1, 4))
    hover_rpm = float(getattr(env, "HOVER_RPM", 14500.0))
    results: List[RoundMetrics] = []

    for r, tgt_xy in enumerate(targets_xy):
        obs, _ = env.reset()
        action[:] = 0.0
        target = np.array([tgt_xy[0], tgt_xy[1], cfg.target_height], dtype=float)
        marker_id = _spawn_target_marker(client, target, color=[0.0, 1.0, 0.0, 1.0])

        start_t = time.time()
        hit_time: Optional[float] = None
        settle_time: Optional[float] = None
        power_samples: List[float] = []
        smoothed_target = np.array([obs[0][0], obs[0][1], cfg.target_height], dtype=float)

        for step in range(max_steps):
            obs, _, terminated, truncated, _ = env.step(action)
            s = obs[0]

            if step < takeoff_steps:
                cmd_target = np.array([s[0], s[1], cfg.target_height], dtype=float)
            else:
                desired = np.array([target[0], target[1], cfg.target_height], dtype=float)
                smoothed_target = 0.82 * smoothed_target + 0.18 * desired
                cmd_target = smoothed_target

            action[0, :], _, _ = controller.computeControlFromState(
                control_timestep=ctrl_dt,
                state=s,
                target_pos=cmd_target,
                target_rpy=np.array([0.0, 0.0, 0.0]),
            )

            dist = float(np.linalg.norm(s[0:3] - target))
            power_samples.append(float(np.sum(np.square(action[0, :] / max(hover_rpm, 1e-6)))))

            if settle_time is None and dist <= cfg.stabilize_dist_m and abs(float(s[2]) - cfg.target_height) <= cfg.stabilize_z_tol_m:
                settle_time = time.time() - start_t

            if hit_time is None and dist <= cfg.hit_radius_m:
                hit_time = time.time() - start_t

            if np.any(terminated) or np.any(truncated):
                break

        p.removeBody(marker_id, physicsClientId=client)
        results.append(
            finalize_round_metrics(r, hit_time, settle_time, power_samples, cfg.round_timeout_sec)
        )

    env.close()
    return results


def _spawn_target_marker(client_id: int, position: np.ndarray, color: List[float]) -> int:
    vis = p.createVisualShape(
        p.GEOM_SPHERE,
        radius=0.05,
        rgbaColor=color,
        physicsClientId=client_id,
    )
    return p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=-1,
        baseVisualShapeIndex=vis,
        basePosition=position.tolist(),
        physicsClientId=client_id,
    )
