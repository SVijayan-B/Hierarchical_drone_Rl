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

    def __init__(self, sim: SimConfig, task: TaskConfig, sensor_cfg: SensorConfig, action_cfg: ActionConfig, use_adaptive_scheduler: bool = True, demo_guided_mode: bool = True, use_mpc_layer: bool = False):
        super().__init__()
        self.sim_cfg = sim
        self.task_cfg = task
        self.sensor_cfg = sensor_cfg
        self.action_cfg = action_cfg
        self.drone_cfg = DroneConfig()

        self.use_adaptive_scheduler = use_adaptive_scheduler
        self.scheduler = AdaptiveGainScheduler()
        self.use_mpc_layer = use_mpc_layer
        if self.use_mpc_layer:
            from hierarchical_drone.controllers.mpc_controller import MPCController
            self.mpc = MPCController(horizon=20, dt=0.1)
        self.current_wind_magnitude = 0.0
        self.control_step_counter = 0
        self.gain_scale = 1.0
        self.pos_history = []
        self.dist_history = []
        self.start_pos = np.zeros(3, dtype=np.float32)

        self.ctrl_dt = 1.0 / self.sim_cfg.ctrl_freq
        self.rl_every_n = max(1, int(self.sim_cfg.ctrl_freq / self.sim_cfg.rl_freq))
        self.max_steps = int(self.sim_cfg.episode_sec * self.sim_cfg.rl_freq)

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
        self.takeoff_steps = int(4.0 * self.sim_cfg.rl_freq)
        self.path_blocked = False
        self.nav_cmd_lpf = np.zeros(3, dtype=np.float32)
        self.guidance_blend = 0.85
        self.demo_guided_mode = demo_guided_mode
        self.camera_follow = True
        self.cam_target = np.array([0.0, 0.0, 0.8], dtype=np.float32)
        self.cam_alpha = 0.10
        self.fall_grace_steps = int(3.0 * self.sim_cfg.rl_freq)
        self.fall_counter = 0
        self.hover_on_target_steps_required = int(3.0 * self.sim_cfg.rl_freq)
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
        if hasattr(self, "debug_line_ids"):
            self._clear_debug_lines()

    def _spawn_target(self):
        # Keep target at a fixed cruise altitude so navigation is mainly XY.
        # Ensure it is far from starting position [0, 0] (between 1.4 and 1.9 meters)
        theta = random.uniform(0.0, 2.0 * np.pi)
        dist = random.uniform(1.4, 1.9)
        self.target = np.array([
            dist * np.cos(theta),
            dist * np.sin(theta),
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
        self.obstacle_ids = []
        self.walls = []
        
        # Start is [0, 0], Target is self.target[0:2]
        tx, ty = self.target[0], self.target[1]
        dist = float(np.linalg.norm(self.target[0:2]))
        if dist < 1e-3:
            return
            
        u = np.array([tx, ty], dtype=np.float32) / dist
        v = np.array([-u[1], u[0]], dtype=np.float32)
        yaw = float(np.arctan2(u[1], u[0]))
        
        # Decide maze offset sign randomly to swap left/right slalom path
        s1 = random.choice([-1.0, 1.0])
        
        # Define the 4 walls: (lambda_along_path, lateral_offset, local_hx, local_hy, yaw_offset)
        wall_specs = [
            (0.28, s1 * 0.35, 0.5, 0.08, np.pi/2),   # Wall 1: Perpendicular gate
            (0.55, -s1 * 0.35, 0.5, 0.08, np.pi/2),  # Wall 2: Perpendicular gate opposite
            (0.82, s1 * 0.35, 0.5, 0.08, np.pi/2),   # Wall 3: Perpendicular gate same as 1
            (0.55, 0.0, 0.4, 0.08, 0.0)              # Wall 4: Parallel center divider
        ]
        
        for idx, (lam, lat, hx, hy, yaw_offset) in enumerate(wall_specs):
            pos_2d = lam * np.array([tx, ty], dtype=np.float32) + lat * dist * v
            hz = random.uniform(0.4, 0.7)
            pos_3d = [float(pos_2d[0]), float(pos_2d[1]), hz]
            
            w_yaw = yaw + yaw_offset
            
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[hx, hy, hz], rgbaColor=[0.7, 0.5, 0.2, 1], physicsClientId=self.client)
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[hx, hy, hz], physicsClientId=self.client)
            orientation = p.getQuaternionFromEuler([0, 0, w_yaw])
            oid = p.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=col,
                baseVisualShapeIndex=vis,
                basePosition=pos_3d,
                baseOrientation=orientation,
                physicsClientId=self.client
            )
            self.obstacle_ids.append(oid)
            self.walls.append((float(pos_2d[0]), float(pos_2d[1]), hx, hy, w_yaw))

    def _get_lookahead_waypoint(self, cur_pos):
        if not hasattr(self, "global_path") or len(self.global_path) == 0:
            return np.array([self.target[0], self.target[1], self.hover_z_ref], dtype=np.float32)
            
        min_dist = float('inf')
        closest_idx = 0
        for i, wp in enumerate(self.global_path):
            d = np.linalg.norm(np.array(wp) - cur_pos[0:2])
            if d < min_dist:
                min_dist = d
                closest_idx = i
                
        lookahead_dist = 0.18
        target_wp_2d = self.global_path[-1]
        for i in range(closest_idx, len(self.global_path)):
            d = np.linalg.norm(np.array(self.global_path[i]) - cur_pos[0:2])
            if d >= lookahead_dist:
                target_wp_2d = self.global_path[i]
                break
                
        return np.array([target_wp_2d[0], target_wp_2d[1], self.hover_z_ref], dtype=np.float32)

    def _clear_debug_lines(self):
        if hasattr(self, "debug_line_ids"):
            for line_id in self.debug_line_ids:
                try:
                    p.removeUserDebugItem(line_id, physicsClientId=self.client)
                except Exception:
                    pass
            self.debug_line_ids = []

    def _draw_path(self):
        self._clear_debug_lines()
        if hasattr(self, "global_path") and len(self.global_path) > 1:
            for i in range(len(self.global_path) - 1):
                p1 = [self.global_path[i][0], self.global_path[i][1], 1.0]
                p2 = [self.global_path[i+1][0], self.global_path[i+1][1], 1.0]
                try:
                    line_id = p.addUserDebugLine(
                        p1, p2,
                        lineColorRGB=[0.0, 0.0, 1.0],  # Blue line
                        lineWidth=4.0,
                        physicsClientId=self.client
                    )
                    self.debug_line_ids.append(line_id)
                except Exception:
                    pass

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
        self.last_state = state[0]
        self._spawn_target()
        self._spawn_obstacles()

        # Initialize dynamic occupancy grid map and planner ( drone is unaware of wall positions initially)
        from hierarchical_drone.utils.a_star import AStarPlanner
        self.planner = AStarPlanner(resolution=0.10, safety_margin=0.18)
        self.mapped_grid = np.zeros((self.planner.nx, self.planner.ny), dtype=np.int8)
        start_pos_2d = [float(state[0][0]), float(state[0][1])]
        target_pos_2d = [float(self.target[0]), float(self.target[1])]
        self.global_path, success = self.planner.plan_on_grid(start_pos_2d, target_pos_2d, self.mapped_grid)
        self.path_blocked = not success
        self.debug_line_ids = []
        self._draw_path()

        self.imu.reset()
        self.prev_action[:] = 0.0
        self.smoothed_action[:] = 0.0
        self.step_count = 0
        self.hover_z_ref = 1.0
        self.clearance_hold_counter = 0
        self.path_blocked = False
        self.pos_ref = np.array(state[0][0:3], dtype=np.float32)
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
        self.vel_history = [np.array(state[0][10:13], dtype=np.float32)]
        self.control_effort_history = []
        self.wp_error_history = []
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

        s = self.last_state
        cur_pos = s[0:3]

        # Read ultrasonic sensors (10 Hz)
        ultra = self.ultra.read(self.drone_id, cur_pos, s[7:10], self.client)

        # Takeoff phase vs. Navigation phase
        if self.step_count <= self.takeoff_steps:
            # During takeoff, hold horizontal position [0,0] and ascend to hover_z_ref (1.0)
            target_wp_astar = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            self.hover_z_ref = 1.0
            vel_cmd = np.zeros(3, dtype=np.float32)
            yaw_rate_cmd = 0.0
            target_wp = target_wp_astar
        else:
            # 1. Dynamic wall climbing logic based on sensors (only if path is blocked)
            # Check horizontal distance to walls
            obs_dist = min(ultra["front"], ultra["left"], ultra["right"], ultra["rear"])
            if self.path_blocked and obs_dist < 0.45:
                # Obstacle detected and path is blocked! Climb over it.
                self.hover_z_ref = 1.55
                self.clearance_hold_counter = 30  # Hold for 3 seconds / 30 steps
            else:
                if self.clearance_hold_counter > 0:
                    self.clearance_hold_counter -= 1
                    self.hover_z_ref = 1.55
                elif self.path_blocked and ultra["down"] < cur_pos[2] - 0.35:
                    # Wall underneath, keep holding
                    self.hover_z_ref = 1.55
                    self.clearance_hold_counter = 10  # Re-hold for 1 second / 10 steps
                else:
                    self.hover_z_ref = 1.0

            # 2. Dynamic mapping & A* path replanning (only outside takeoff phase)
            need_replan = False
            yaw = s[9]
            c_y, s_y = np.cos(yaw), np.sin(yaw)
            rot_z = np.array([[c_y, -s_y, 0.0], [s_y, c_y, 0.0], [0.0, 0.0, 1.0]])
            
            dirs_body = {
                "front": np.array([1.0, 0.0, 0.0]),
                "left": np.array([0.0, 1.0, 0.0]),
                "right": np.array([0.0, -1.0, 0.0]),
                "rear": np.array([-1.0, 0.0, 0.0]),
            }
            
            for name, d_body in dirs_body.items():
                dist = ultra[name]
                if dist < self.sensor_cfg.ultrasonic_max_range * 0.95:
                    d_world = rot_z @ d_body
                    obs_pos = cur_pos + np.array([0.0, 0.0, 0.03]) + d_world * dist
                    
                    gx, gy = self.planner._world_to_grid(obs_pos[0], obs_pos[1])
                    
                    # Inflate wall obstacle cell
                    inflation_cells = int(self.planner.safety_margin / self.planner.resolution)
                    for dx in range(-inflation_cells, inflation_cells + 1):
                        for dy in range(-inflation_cells, inflation_cells + 1):
                            ngx = gx + dx
                            ngy = gy + dy
                            if 0 <= ngx < self.planner.nx and 0 <= ngy < self.planner.ny:
                                if self.mapped_grid[ngx, ngy] == 0:
                                    self.mapped_grid[ngx, ngy] = 1
                                    need_replan = True

            # If new obstacles mapped, or first step after takeoff, replan path
            if need_replan or self.step_count == self.takeoff_steps + 1:
                self.global_path, success = self.planner.plan_on_grid(cur_pos[0:2], self.target[0:2], self.mapped_grid)
                self.path_blocked = not success
                self._draw_path()

            # Get lookahead waypoint from A* path
            target_wp_astar = self._get_lookahead_waypoint(cur_pos)

            # 3. Determine target_wp (if MPC) or vel_cmd (if non-MPC), and yaw_rate_cmd
            if self.use_mpc_layer:
                if self.demo_guided_mode:
                    target_wp = target_wp_astar
                    yaw_rate_cmd = 0.0
                    vel_cmd = np.zeros(3, dtype=np.float32)
                else:
                    clipped = np.clip(action, -1.0, 1.0)
                    delta = clipped - self.smoothed_action
                    delta = np.clip(delta, -self.action_cfg.action_rate_limit, self.action_cfg.action_rate_limit)
                    clipped = self.smoothed_action + delta
                    self.smoothed_action = (1.0 - self.action_cfg.action_smoothing_alpha) * self.smoothed_action + self.action_cfg.action_smoothing_alpha * clipped
                    
                    # Local waypoint: drone position + offset (up to 0.5m)
                    wp_offset = self.smoothed_action[0:3] * 0.5
                    rl_wp = cur_pos + wp_offset
                    
                    # Blend local waypoint and lookahead waypoint (85% lookahead, 15% local waypoint)
                    target_wp = (1.0 - self.guidance_blend) * rl_wp + self.guidance_blend * target_wp_astar
                    yaw_rate_cmd = self.smoothed_action[3] * self.action_cfg.yaw_rate_max
                    
                    # Define vel_cmd for reward calculation
                    cur_vel = s[10:13]
                    vel_cmd = self.mpc.compute_control(cur_pos, cur_vel, target_wp)
            else:
                vel_cmd, yaw_rate_cmd = self._action_to_vel_cmd(action)

        motor_action = np.ones((1, 4), dtype=np.float32) * self.drone_cfg.hover_rpm

        for _ in range(self.rl_every_n):
            self.control_step_counter += 1
            self._apply_wind_disturbance()
            state, _, terminated, truncated, _ = self.env.step(motor_action)
            s = state[0]
            self.last_state = s

            # Record state histories for metrics
            self.pos_history.append(np.array(s[0:3], dtype=np.float32))
            rel = self.target - s[0:3]
            self.dist_history.append(float(np.linalg.norm(rel)))
            self.vel_history.append(np.array(s[10:13], dtype=np.float32))

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

            # 2. Get inner step targets
            if self.step_count <= self.takeoff_steps:
                target_wp_astar_inner = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            else:
                target_wp_astar_inner = self._get_lookahead_waypoint(s[0:3])

            # Compute commanded velocity mixed_vel
            if self.use_mpc_layer:
                cur_pos_inner = s[0:3]
                cur_vel_inner = s[10:13]
                if not self.demo_guided_mode and self.step_count > self.takeoff_steps:
                    rl_wp = cur_pos_inner + wp_offset
                    target_wp_inner = (1.0 - self.guidance_blend) * rl_wp + self.guidance_blend * target_wp_astar_inner
                else:
                    target_wp_inner = target_wp_astar_inner
                
                # MPC controller computes desired velocity towards target_wp_inner
                vel_cmd_mpc = self.mpc.compute_control(cur_pos_inner, cur_vel_inner, target_wp_inner)
                
                # Smooth startup ramp
                ramp = float(np.clip(self.step_count / max(1, self.takeoff_steps), 0.10, 1.0))
                mixed_vel = vel_cmd_mpc * ramp
            else:
                # Diagonal guidance: move in XY while climbing towards target_wp_astar_inner
                if self.step_count <= self.takeoff_steps:
                    rel_inner = target_wp_astar_inner - s[0:3]
                    dist3 = float(np.linalg.norm(rel_inner))
                    if dist3 > 1e-6:
                        guide_dir = rel_inner / dist3
                    else:
                        guide_dir = np.zeros(3, dtype=np.float32)
                    guide_speed = min(self.action_cfg.vz_max, max(0.05, 0.30 * dist3))
                    guide_vel = guide_dir * guide_speed
                else:
                    rel_inner = target_wp_astar_inner - s[0:3]
                    dist3 = float(np.linalg.norm(rel_inner))
                    if dist3 > 1e-6:
                        guide_dir = rel_inner / dist3
                    else:
                        guide_dir = np.zeros(3, dtype=np.float32)

                    guide_speed = min(self.action_cfg.vxy_max, max(0.06, 0.20 * dist3))
                    guide_vel = guide_dir * guide_speed
                    guide_vel[2] = float(np.clip(guide_vel[2], -self.action_cfg.vz_max, self.action_cfg.vz_max))

                # Smooth startup ramp for stable emergence from ground.
                ramp = float(np.clip(self.step_count / max(1, self.takeoff_steps), 0.10, 1.0))
                mixed_vel = (1.0 - self.guidance_blend) * vel_cmd + self.guidance_blend * guide_vel
                mixed_vel *= ramp

            # If climbing over a wall and not yet at safe altitude, stop horizontal motion
            if self.hover_z_ref > 1.05 and s[2] < 1.45:
                mixed_vel[0] = 0.0
                mixed_vel[1] = 0.0

            # Local obstacle repulsion (potential field) to prevent wall collisions
            repulse_vel = np.zeros(3, dtype=np.float32)
            repulse_dist = 0.14
            k_rep = 0.35
            
            yaw = s[9]
            c_y, s_y = np.cos(yaw), np.sin(yaw)
            
            rep_body = np.zeros(2, dtype=np.float32)
            if ultra["front"] < repulse_dist:
                rep_body[0] -= k_rep * (repulse_dist - ultra["front"])
            if ultra["rear"] < repulse_dist:
                rep_body[0] += k_rep * (repulse_dist - ultra["rear"])
            if ultra["left"] < repulse_dist:
                rep_body[1] -= k_rep * (repulse_dist - ultra["left"])
            if ultra["right"] < repulse_dist:
                rep_body[1] += k_rep * (repulse_dist - ultra["right"])
                
            repulse_vel[0] = rep_body[0] * c_y - rep_body[1] * s_y
            repulse_vel[1] = rep_body[0] * s_y + rep_body[1] * c_y
            
            mixed_vel += repulse_vel
            
            # Ensure velocity limits are respected after adding repulsion
            mixed_vel[0] = np.clip(mixed_vel[0], -self.action_cfg.vxy_max, self.action_cfg.vxy_max)
            mixed_vel[1] = np.clip(mixed_vel[1], -self.action_cfg.vxy_max, self.action_cfg.vxy_max)
            mixed_vel[2] = np.clip(mixed_vel[2], -self.action_cfg.vz_max, self.action_cfg.vz_max)

            self.nav_cmd_lpf = 0.95 * self.nav_cmd_lpf + 0.05 * mixed_vel
            
            target_vel = np.array([self.nav_cmd_lpf[0], self.nav_cmd_lpf[1], self.nav_cmd_lpf[2]], dtype=np.float32)
            
            # Integrate target_vel to get a smooth reference position target_pos
            self.pos_ref = self.pos_ref + target_vel * self.ctrl_dt
            
            # Anti-windup reference clipping: prevent it from running too far ahead of the drone
            diff_ref = self.pos_ref - s[0:3]
            dist_ref = float(np.linalg.norm(diff_ref))
            if dist_ref > 0.15:
                self.pos_ref = s[0:3] + diff_ref * (0.15 / dist_ref)
                
            target_pos = np.copy(self.pos_ref)
            target_yaw = s[9]
            target_rpy_rates = np.array([0.0, 0.0, yaw_rate_cmd], dtype=np.float32)
            
            rpm, _, _ = self.stab_ctrl.computeControlFromState(
                control_timestep=self.ctrl_dt,
                state=s,
                target_pos=target_pos,
                target_rpy=np.array([0.0, 0.0, target_yaw], dtype=np.float32),
                target_vel=target_vel,
                target_rpy_rates=target_rpy_rates,
            )
            motor_action[0, :] = np.clip(rpm, self.drone_cfg.min_rpm, self.drone_cfg.max_rpm)

            # Record control step metrics
            self.control_effort_history.append(np.sum(np.square(target_vel)))
            
            if self.use_mpc_layer:
                if self.step_count <= self.takeoff_steps:
                    wp_err = float(np.linalg.norm(s[0:3] - target_wp))
                else:
                    wp_err = float(np.linalg.norm(s[0:3] - target_wp_inner))
            else:
                wp_err = float(np.linalg.norm(s[0:3] - target_pos))
            self.wp_error_history.append(wp_err)

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

        progress_reward = 5.0 * (self.prev_dist - dist_xy)
        target_bonus = 14.0 if dist_xy < self.task_cfg.target_threshold_m else 0.0
        proximity_pen = -0.8 * max(0.0, 0.30 - min(ultra["front"], ultra["left"], ultra["right"], ultra["rear"]))
        tilt_pen = -0.10 * (abs(rpy[0]) + abs(rpy[1]))
        rate_pen = -0.015 * np.linalg.norm(rates)
        smooth_pen = -0.03 * np.linalg.norm(self.smoothed_action - self.prev_action)
        effort_pen = -0.01 * np.linalg.norm(vel_cmd)
        vel_stability = -0.015 * np.linalg.norm(vel)

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
            reward -= 0.25
        else:
            self.fall_counter = 0

        done = success or out_of_bounds or (self.step_count >= self.max_steps) or (self.fall_counter >= self.fall_grace_steps)
        if collision:
            reward -= 2.0
        if out_of_bounds:
            reward -= 6.0

        self.prev_dist = dist_xy
        info: Dict[str, float] = {
            "distance": dist,
            "distance_xy": dist_xy,
            "collision": float(collision),
            "fell_down": float(fell_down),
            "hover_on_target_sec": float(self.hover_on_target_counter / self.sim_cfg.rl_freq),
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
                settling_time = float(self.max_steps * (self.rl_every_n * self.ctrl_dt))
            else:
                settling_time = float(settled_idx * self.ctrl_dt)

            info["mean_tracking_error"] = mean_tracking_err
            info["rmse_tracking_error"] = rmse_tracking_err
            info["overshoot"] = overshoot
            info["overshoot_percent"] = overshoot_pct
            info["z_overshoot"] = z_overshoot
            info["settling_time"] = settling_time

            # Compute new metrics
            vels = np.array(self.vel_history)
            if len(vels) > 2:
                accels = np.diff(vels, axis=0) / self.ctrl_dt
                jerks = np.diff(accels, axis=0) / self.ctrl_dt
                smoothness = float(np.sqrt(np.mean(np.square(jerks))))
            else:
                smoothness = 0.0
            info["trajectory_smoothness"] = smoothness

            # 2. Control Effort
            info["control_effort"] = float(np.mean(self.control_effort_history)) if self.control_effort_history else 0.0

            # 3. Waypoint Tracking Error
            info["waypoint_tracking_error"] = float(np.mean(self.wp_error_history)) if self.wp_error_history else 0.0

        return obs, float(reward), done, False, info

    def close(self):
        self._clear_scene()
        self.env.close()
