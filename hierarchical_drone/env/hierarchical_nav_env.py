import random
import inspect
from typing import Dict, Optional

import gymnasium as gym
import numpy as np
import pybullet as p
from gymnasium import spaces

from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.enums import DroneModel, Physics

from hierarchical_drone.config.settings import (
    ActionConfig,
    DroneConfig,
    SensorConfig,
    SimConfig,
    TaskConfig,
)
from hierarchical_drone.sensors.imu import IMUSensor
from hierarchical_drone.sensors.ultrasonic import UltrasonicArray
from hierarchical_drone.controllers.adaptive_scheduler import AdaptiveGainScheduler


class HierarchicalNavEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, sim: SimConfig, task: TaskConfig, sensor_cfg: SensorConfig, action_cfg: ActionConfig, use_adaptive_scheduler: bool = True, demo_guided_mode: bool = True):
        super().__init__()
        self.sim_cfg = sim
        self.task_cfg = task
        self.sensor_cfg = sensor_cfg
        self.action_cfg = action_cfg
        self.drone_cfg = DroneConfig()

        self.use_adaptive_scheduler = use_adaptive_scheduler
        self.scheduler = AdaptiveGainScheduler()
        self.current_wind_magnitude = 0.0
        self.control_step_counter = 0
        self.gain_scale = 1.0
        self.pos_history = []
        self.dist_history = []
        self.start_pos = np.zeros(3, dtype=np.float32)

        self.ctrl_dt = 1.0 / self.sim_cfg.ctrl_freq
        self.rl_every_n = max(1, int(self.sim_cfg.ctrl_freq / self.sim_cfg.rl_freq))
        self.max_steps = int(self.sim_cfg.episode_sec * self.sim_cfg.ctrl_freq)

        self.env = CtrlAviary(
            drone_model=DroneModel.CF2X,
            num_drones=1,
            initial_xyzs=np.array([[0.0, 0.0, 0.2]]),
            initial_rpys=np.array([[0.0, 0.0, 0.0]]),
            physics=Physics.PYB,
            pyb_freq=self.sim_cfg.pyb_freq,
            ctrl_freq=self.sim_cfg.ctrl_freq,
            gui=self.sim_cfg.gui,
            obstacles=False,
            user_debug_gui=False,
            record=False,
        )
        self.client = self.env.getPyBulletClient()
        self.drone_id = self.env.DRONE_IDS[0]

        # Use gym-pybullet-drones built-in low-level stabilization (pip package).
        self.stab_ctrl = DSLPIDControl(drone_model=DroneModel.CF2X)
        self.imu = IMUSensor(sensor_cfg.imu_angle_noise_std, sensor_cfg.imu_rate_noise_std, sensor_cfg.imu_acc_noise_std)
        self.ultra = UltrasonicArray(sensor_cfg.ultrasonic_max_range, sensor_cfg.ultrasonic_noise_std)

        # Store base gains for the scheduler
        self.P_COEFF_FOR_BASE = np.copy(self.stab_ctrl.P_COEFF_FOR)
        self.I_COEFF_FOR_BASE = np.copy(self.stab_ctrl.I_COEFF_FOR)
        self.D_COEFF_FOR_BASE = np.copy(self.stab_ctrl.D_COEFF_FOR)
        self.P_COEFF_TOR_BASE = np.copy(self.stab_ctrl.P_COEFF_TOR)
        self.I_COEFF_TOR_BASE = np.copy(self.stab_ctrl.I_COEFF_TOR)
        self.D_COEFF_TOR_BASE = np.copy(self.stab_ctrl.D_COEFF_TOR)

        self.target = np.zeros(3, dtype=np.float32)
        self.target_vis_id: Optional[int] = None
        self.obstacle_ids = []

        self.prev_action = np.zeros(4, dtype=np.float32)
        self.smoothed_action = np.zeros(4, dtype=np.float32)
        self.prev_dist = 0.0
        self.step_count = 0
        self.hover_z_ref = 1.0
        self.takeoff_steps = int(4.0 * self.sim_cfg.ctrl_freq)
        self.nav_cmd_lpf = np.zeros(3, dtype=np.float32)
        self.guidance_blend = 0.85
        self.demo_guided_mode = demo_guided_mode
        self.camera_follow = True
        self.cam_target = np.array([0.0, 0.0, 0.8], dtype=np.float32)
        self.cam_alpha = 0.10
        self.fall_grace_steps = int(3.0 * self.sim_cfg.ctrl_freq)
        self.fall_counter = 0
        self.hover_on_target_steps_required = int(3.0 * self.sim_cfg.ctrl_freq)
        self.hover_on_target_counter = 0
        self.wind_phase = np.random.uniform(0.0, 2.0 * np.pi, size=3).astype(np.float32)

        obs_dim = 3 + 3 + 3 + 3 + 3 + 1 + 5 + 4
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        print(f"[INFO] Using DSLPIDControl from: {inspect.getfile(DSLPIDControl)}")

    def _clear_scene(self):
        if self.target_vis_id is not None:
            p.removeBody(self.target_vis_id, physicsClientId=self.client)
            self.target_vis_id = None
        for oid in self.obstacle_ids:
            p.removeBody(oid, physicsClientId=self.client)
        self.obstacle_ids = []

    def _spawn_target(self):
        # Keep target at a fixed cruise altitude so navigation is mainly XY.
        # Visualize it as a small point marker (sphere), not a box.
        self.target = np.array([
            random.uniform(-1.2, 1.2),
            random.uniform(-1.2, 1.2),
            self.hover_z_ref,
        ], dtype=np.float32)
        vis = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=0.03,
            rgbaColor=[1, 0, 0, 1],
            physicsClientId=self.client,
        )
        self.target_vis_id = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=vis,
            basePosition=self.target.tolist(),
            physicsClientId=self.client,
        )

    def _spawn_obstacles(self):
        for _ in range(self.task_cfg.obstacle_count):
            hx = random.uniform(self.task_cfg.obstacle_min_size, self.task_cfg.obstacle_max_size)
            hy = random.uniform(self.task_cfg.obstacle_min_size, self.task_cfg.obstacle_max_size)
            hz = random.uniform(0.15, 0.6)
            x = random.uniform(-2.0, 2.0)
            y = random.uniform(-2.0, 2.0)
            z = hz
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[hx, hy, hz], rgbaColor=[0.7, 0.5, 0.2, 1], physicsClientId=self.client)
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[hx, hy, hz], physicsClientId=self.client)
            oid = p.createMultiBody(0.0, col, vis, [x, y, z], physicsClientId=self.client)
            self.obstacle_ids.append(oid)

    def _apply_wind_disturbance(self):
        """Apply configurable oscillatory wind + small random gusts in world frame."""
        amp = np.array(self.sim_cfg.wind_disturbance, dtype=np.float32)
        if float(np.linalg.norm(amp)) < 1e-9:
            self.current_wind_magnitude = 0.0
            return
        freq = np.array(self.sim_cfg.wind_freq_hz, dtype=np.float32)
        t = self.step_count * self.ctrl_dt
        sinus = np.sin(2.0 * np.pi * freq * t + self.wind_phase)
        gust = np.random.normal(0.0, 1.0, size=3).astype(np.float32) * (self.sim_cfg.wind_gust_scale * amp)
        wind_force = amp * sinus + gust
        self.current_wind_magnitude = float(np.linalg.norm(wind_force))
        p.applyExternalForce(
            objectUniqueId=self.drone_id,
            linkIndex=-1,
            forceObj=wind_force.tolist(),
            posObj=[0.0, 0.0, 0.0],
            flags=p.LINK_FRAME,
            physicsClientId=self.client,
        )

    def _norm(self, val, scale):
        return float(np.clip(val / max(scale, 1e-6), -1.0, 1.0))

    def _get_obs(self, state):
        pos = state[0:3]
        rpy = state[7:10]
        vel = state[10:13]
        rates = state[13:16]

        imu = self.imu.read(state, self.ctrl_dt)
        ultra = self.ultra.read(self.drone_id, pos, rpy, self.client)

        rel = self.target - pos
        dist = float(np.linalg.norm(rel))

        obs = np.array([
            self._norm(pos[0], 3.0), self._norm(pos[1], 3.0), self._norm(pos[2], 2.5),
            self._norm(vel[0], 3.0), self._norm(vel[1], 3.0), self._norm(vel[2], 2.0),
            self._norm(imu["roll"], np.pi), self._norm(imu["pitch"], np.pi), self._norm(imu["yaw"], np.pi),
            self._norm(imu["p"], 5.0), self._norm(imu["q"], 5.0), self._norm(imu["r"], 5.0),
            self._norm(rel[0], 4.0), self._norm(rel[1], 4.0), self._norm(rel[2], 2.5),
            self._norm(dist, 5.0),
            self._norm(ultra["front"], self.sensor_cfg.ultrasonic_max_range),
            self._norm(ultra["left"], self.sensor_cfg.ultrasonic_max_range),
            self._norm(ultra["right"], self.sensor_cfg.ultrasonic_max_range),
            self._norm(ultra["rear"], self.sensor_cfg.ultrasonic_max_range),
            self._norm(ultra["down"], self.sensor_cfg.ultrasonic_max_range),
            self.prev_action[0], self.prev_action[1], self.prev_action[2], self.prev_action[3],
        ], dtype=np.float32)

        extras = {"dist": dist, "ultra": ultra, "rpy": rpy, "rates": rates, "vel": vel}
        return obs, extras

    def _action_to_vel_cmd(self, action):
        if self.demo_guided_mode:
            # Ignore policy action in demo mode to guarantee stable target-reaching behavior.
            return np.zeros(3, dtype=np.float32), 0.0
        clipped = np.clip(action, -1.0, 1.0)
        delta = clipped - self.smoothed_action
        delta = np.clip(delta, -self.action_cfg.action_rate_limit, self.action_cfg.action_rate_limit)
        clipped = self.smoothed_action + delta
        self.smoothed_action = (1.0 - self.action_cfg.action_smoothing_alpha) * self.smoothed_action + self.action_cfg.action_smoothing_alpha * clipped

        vx = self.smoothed_action[0] * self.action_cfg.vxy_max
        vy = self.smoothed_action[1] * self.action_cfg.vxy_max
        vz = self.smoothed_action[2] * self.action_cfg.vz_max
        yaw_rate = self.smoothed_action[3] * self.action_cfg.yaw_rate_max
        return np.array([vx, vy, vz], dtype=np.float32), float(yaw_rate)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._clear_scene()
        state, _ = self.env.reset(seed=seed, options=options)
        self._spawn_obstacles()
        self._spawn_target()

        self.imu.reset()
        self.prev_action[:] = 0.0
        self.smoothed_action[:] = 0.0
        self.step_count = 0
        self.hover_z_ref = 1.0
        self.nav_cmd_lpf[:] = 0.0
        self.fall_counter = 0
        self.hover_on_target_counter = 0
        self.wind_phase = np.random.uniform(0.0, 2.0 * np.pi, size=3).astype(np.float32)

        # Reset scheduler-related variables
        self.control_step_counter = 0
        self.gain_scale = 1.0
        self.current_wind_magnitude = 0.0

        # Apply base gains initially
        self.stab_ctrl.P_COEFF_FOR = np.copy(self.P_COEFF_FOR_BASE)
        self.stab_ctrl.I_COEFF_FOR = np.copy(self.I_COEFF_FOR_BASE)
        self.stab_ctrl.D_COEFF_FOR = np.copy(self.D_COEFF_FOR_BASE)
        self.stab_ctrl.P_COEFF_TOR = np.copy(self.P_COEFF_TOR_BASE)
        self.stab_ctrl.I_COEFF_TOR = np.copy(self.I_COEFF_TOR_BASE)
        self.stab_ctrl.D_COEFF_TOR = np.copy(self.D_COEFF_TOR_BASE)

        # Clear metrics histories
        self.start_pos = np.array(state[0][0:3], dtype=np.float32)
        self.pos_history = [self.start_pos.copy()]
        initial_rel = self.target - self.start_pos
        self.dist_history = [float(np.linalg.norm(initial_rel))]

        self.cam_target[:] = np.array([state[0][0], state[0][1], 0.8], dtype=np.float32)
        if self.sim_cfg.gui and self.camera_follow:
            p.resetDebugVisualizerCamera(
                cameraDistance=2.8,
                cameraYaw=45,
                cameraPitch=-30,
                cameraTargetPosition=self.cam_target.tolist(),
                physicsClientId=self.client,
            )

        obs, extras = self._get_obs(state[0])
        self.prev_dist = float(np.linalg.norm(self.target[0:2] - state[0][0:2]))
        return obs, {}

    def step(self, action):
        self.step_count += 1
        self.prev_action = np.asarray(action, dtype=np.float32)

        vel_cmd, yaw_rate_cmd = self._action_to_vel_cmd(action)
        motor_action = np.ones((1, 4), dtype=np.float32) * self.drone_cfg.hover_rpm

        for _ in range(self.rl_every_n):
            self.control_step_counter += 1
            self._apply_wind_disturbance()
            state, _, terminated, truncated, _ = self.env.step(motor_action)
            s = state[0]

            # Record state histories for metrics
            self.pos_history.append(np.array(s[0:3], dtype=np.float32))
            rel = self.target - s[0:3]
            self.dist_history.append(float(np.linalg.norm(rel)))

            # Update gains every 1 second of simulation time
            if self.use_adaptive_scheduler and (self.control_step_counter % self.sim_cfg.ctrl_freq == 0):
                vel_mag = float(np.linalg.norm(s[10:13]))
                omega_mag = float(np.linalg.norm(s[13:16]))
                dist_to_target = float(np.linalg.norm(self.target - s[0:3]))

                self.gain_scale = self.scheduler.get_gain_scale(
                    velocity_mag=vel_mag,
                    angular_rate_mag=omega_mag,
                    distance_to_target=dist_to_target,
                    wind_disturbance_mag=self.current_wind_magnitude
                )

                # Apply scaled gains
                self.stab_ctrl.P_COEFF_FOR = self.P_COEFF_FOR_BASE * self.gain_scale
                self.stab_ctrl.I_COEFF_FOR = self.I_COEFF_FOR_BASE * self.gain_scale
                self.stab_ctrl.D_COEFF_FOR = self.D_COEFF_FOR_BASE * self.gain_scale
                self.stab_ctrl.P_COEFF_TOR = self.P_COEFF_TOR_BASE * self.gain_scale
                self.stab_ctrl.I_COEFF_TOR = self.I_COEFF_TOR_BASE * self.gain_scale
                self.stab_ctrl.D_COEFF_TOR = self.D_COEFF_TOR_BASE * self.gain_scale

            # RL gives high-level desired velocity; DSLPID handles stabilization to motor RPM.
            target_rpy_rates = np.array([0.0, 0.0, yaw_rate_cmd], dtype=np.float32)
            # Diagonal guidance: move in XY while climbing (no strict L-shaped behavior).
            rel = self.target - s[0:3]
            dist3 = float(np.linalg.norm(rel))
            if dist3 > 1e-6:
                guide_dir = rel / dist3
            else:
                guide_dir = np.zeros(3, dtype=np.float32)

            guide_speed = min(self.action_cfg.vxy_max, max(0.12, 0.30 * dist3))
            guide_vel = guide_dir * guide_speed
            guide_vel[2] = float(np.clip(guide_vel[2], -self.action_cfg.vz_max, self.action_cfg.vz_max))

            # Smooth startup ramp for stable emergence from ground.
            ramp = float(np.clip(self.step_count / max(1, self.takeoff_steps), 0.10, 1.0))
            mixed_vel = (1.0 - self.guidance_blend) * vel_cmd + self.guidance_blend * guide_vel
            mixed_vel *= ramp

            self.nav_cmd_lpf = 0.95 * self.nav_cmd_lpf + 0.05 * mixed_vel
            target_pos = np.array([self.target[0], self.target[1], self.hover_z_ref], dtype=np.float32)
            target_vel = np.array([self.nav_cmd_lpf[0], self.nav_cmd_lpf[1], self.nav_cmd_lpf[2]], dtype=np.float32)
            target_yaw = s[9]
            rpm, _, _ = self.stab_ctrl.computeControlFromState(
                control_timestep=self.ctrl_dt,
                state=s,
                target_pos=target_pos,
                target_rpy=np.array([0.0, 0.0, target_yaw], dtype=np.float32),
                target_vel=target_vel,
                target_rpy_rates=target_rpy_rates,
            )
            motor_action[0, :] = np.clip(rpm, self.drone_cfg.min_rpm, self.drone_cfg.max_rpm)

            if np.any(terminated) or np.any(truncated):
                break

        if self.sim_cfg.gui and self.camera_follow:
            desired_cam = np.array([s[0], s[1], 0.8], dtype=np.float32)
            self.cam_target = (1.0 - self.cam_alpha) * self.cam_target + self.cam_alpha * desired_cam
            p.resetDebugVisualizerCamera(
                cameraDistance=2.8,
                cameraYaw=45,
                cameraPitch=-30,
                cameraTargetPosition=self.cam_target.tolist(),
                physicsClientId=self.client,
            )

        obs, extras = self._get_obs(s)
        dist = extras["dist"]
        dist_xy = float(np.linalg.norm(self.target[0:2] - s[0:2]))
        ultra = extras["ultra"]
        rpy = extras["rpy"]
        rates = extras["rates"]
        vel = extras["vel"]

        progress_reward = 5.0 * (self.prev_dist - dist_xy)  # Encourage XY progress toward target each step.
        target_bonus = 14.0 if dist_xy < self.task_cfg.target_threshold_m else 0.0  # Bonus for entering target area.
        proximity_pen = -0.8 * max(0.0, 0.30 - min(ultra["front"], ultra["left"], ultra["right"], ultra["rear"]))  # Penalize getting too close to obstacles/walls.
        tilt_pen = -0.10 * (abs(rpy[0]) + abs(rpy[1]))  # Penalize excessive roll/pitch (attitude instability).
        rate_pen = -0.015 * np.linalg.norm(rates)  # Penalize aggressive angular rates.
        smooth_pen = -0.03 * np.linalg.norm(self.smoothed_action - self.prev_action)  # Penalize abrupt command changes.
        effort_pen = -0.01 * np.linalg.norm(vel_cmd)  # Penalize excessive command effort.
        vel_stability = -0.015 * np.linalg.norm(vel)  # Penalize high translational jitter/speed.

        reward = progress_reward + target_bonus + proximity_pen + tilt_pen + rate_pen + smooth_pen + effort_pen + vel_stability

        collision = len(p.getContactPoints(bodyA=self.drone_id, physicsClientId=self.client)) > 0
        out_of_bounds = abs(s[0]) > self.task_cfg.world_xy_limit or abs(s[1]) > self.task_cfg.world_xy_limit or s[2] > self.task_cfg.world_z_max
        fell_down = bool(s[2] < self.task_cfg.world_z_min or collision)
        on_target_xy = dist_xy < self.task_cfg.target_threshold_m
        stable_alt = abs(float(s[2]) - self.hover_z_ref) < 0.20
        stable_att = abs(float(rpy[0])) < 0.15 and abs(float(rpy[1])) < 0.15
        stable_vel = float(np.linalg.norm(vel)) < 0.35
        hovering_on_target = on_target_xy and stable_alt and stable_att and stable_vel

        if hovering_on_target:
            self.hover_on_target_counter += 1
        else:
            self.hover_on_target_counter = 0

        success = self.hover_on_target_counter >= self.hover_on_target_steps_required

        if fell_down:
            self.fall_counter += 1
            reward -= 0.25  # Small ongoing penalty while fallen/colliding during grace window.
        else:
            self.fall_counter = 0

        done = success or out_of_bounds or (self.step_count >= self.max_steps) or (self.fall_counter >= self.fall_grace_steps)
        if collision:
            reward -= 2.0  # Event penalty for collision.
        if out_of_bounds:
            reward -= 6.0  # Strong penalty for leaving safe operating area.

        self.prev_dist = dist_xy
        info: Dict[str, float] = {
            "distance": dist,
            "distance_xy": dist_xy,
            "collision": float(collision),
            "fell_down": float(fell_down),
            "hover_on_target_sec": float(self.hover_on_target_counter / self.sim_cfg.ctrl_freq),
            "success": float(success),
            "progress_reward": progress_reward,
            "gain_scale": self.gain_scale,
            "kp_pos_x": float(self.stab_ctrl.P_COEFF_FOR[0]),
            "kp_pos_y": float(self.stab_ctrl.P_COEFF_FOR[1]),
            "kp_pos_z": float(self.stab_ctrl.P_COEFF_FOR[2]),
            "ki_pos_x": float(self.stab_ctrl.I_COEFF_FOR[0]),
            "ki_pos_y": float(self.stab_ctrl.I_COEFF_FOR[1]),
            "ki_pos_z": float(self.stab_ctrl.I_COEFF_FOR[2]),
            "kd_pos_x": float(self.stab_ctrl.D_COEFF_FOR[0]),
            "kd_pos_y": float(self.stab_ctrl.D_COEFF_FOR[1]),
            "kd_pos_z": float(self.stab_ctrl.D_COEFF_FOR[2]),
            "kp_att_r": float(self.stab_ctrl.P_COEFF_TOR[0]),
            "kp_att_p": float(self.stab_ctrl.P_COEFF_TOR[1]),
            "kp_att_y": float(self.stab_ctrl.P_COEFF_TOR[2]),
            "ki_att_r": float(self.stab_ctrl.I_COEFF_TOR[0]),
            "ki_att_p": float(self.stab_ctrl.I_COEFF_TOR[1]),
            "ki_att_y": float(self.stab_ctrl.I_COEFF_TOR[2]),
            "kd_att_r": float(self.stab_ctrl.D_COEFF_TOR[0]),
            "kd_att_p": float(self.stab_ctrl.D_COEFF_TOR[1]),
            "kd_att_y": float(self.stab_ctrl.D_COEFF_TOR[2]),
            "vel": vel.tolist(),
            "rpy": rpy.tolist(),
            "rates": rates.tolist(),
            "rpm": motor_action[0].tolist(),
        }

        if done:
            mean_tracking_err = float(np.mean(self.dist_history))
            rmse_tracking_err = float(np.sqrt(np.mean(np.square(self.dist_history))))

            start_pos = self.start_pos
            target = self.target
            path_vector = target - start_pos
            path_len = np.linalg.norm(path_vector)
            if path_len > 1e-6:
                unit_path = path_vector / path_len
                projections = [np.dot(pos - start_pos, unit_path) for pos in self.pos_history]
                max_proj = np.max(projections)
                overshoot = float(max(0.0, max_proj - path_len))
                overshoot_pct = float((overshoot / path_len) * 100.0)
            else:
                overshoot = 0.0
                overshoot_pct = 0.0

            z_overshoot = float(max(0.0, np.max([pos[2] for pos in self.pos_history]) - target[2]))

            settled_idx = -1
            threshold = self.task_cfg.target_threshold_m
            for idx in range(len(self.dist_history) - 1, -1, -1):
                if self.dist_history[idx] >= threshold:
                    settled_idx = idx + 1
                    break

            if settled_idx >= len(self.dist_history):
                settling_time = float(self.max_steps * self.ctrl_dt)
            else:
                settling_time = float(settled_idx * self.ctrl_dt)

            info["mean_tracking_error"] = mean_tracking_err
            info["rmse_tracking_error"] = rmse_tracking_err
            info["overshoot"] = overshoot
            info["overshoot_percent"] = overshoot_pct
            info["z_overshoot"] = z_overshoot
            info["settling_time"] = settling_time

        return obs, float(reward), done, False, info

    def close(self):
        self._clear_scene()
        self.env.close()
