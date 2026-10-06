import numpy as np


class MPCController:
    """Model Predictive Controller for 3D trajectory tracking with velocity and acceleration constraints.
    
    Upgraded to support Adaptive MPC with dynamic horizons (10, 20, 30) and adaptive cost weights.
    """

    def __init__(self, horizon: int = 20, dt: float = 0.1):
        self.dt = dt
        self.N = horizon  # default nominal horizon

        # State-space formulation:
        # State z = [x, y, z, vx, vy, vz]^T  (6D)
        # Input u = [ax, ay, az]^T          (3D)
        # Continuous-time:
        # x_dot = v, v_dot = u
        # Discrete-time state transition matrices A and B:
        # z_{k+1} = A*z_k + B*u_k
        self.A = np.eye(6)
        self.A[0:3, 3:6] = np.eye(3) * self.dt

        self.B = np.zeros((6, 3))
        self.B[0:3, 0:3] = 0.5 * (self.dt**2) * np.eye(3)
        self.B[3:6, 0:3] = self.dt * np.eye(3)

        # Precompute prediction matrices M and C for horizons 10, 20, and 30
        self.M_dict = {}
        self.C_dict = {}
        for h in [10, 20, 30]:
            M_h = []
            for i in range(1, h + 1):
                M_h.append(np.linalg.matrix_power(self.A, i))
            self.M_dict[h] = np.vstack(M_h)  # Shape (6h, 6)

            C_h = np.zeros((6 * h, 3 * h))
            for r in range(h):
                for c in range(r + 1):
                    power = r - c
                    C_h[r * 6 : (r + 1) * 6, c * 3 : (c + 1) * 3] = np.linalg.matrix_power(self.A, power) @ self.B
            self.C_dict[h] = C_h

        # Performance constraints
        self.vxy_max = 0.08
        self.vz_max = 0.08
        self.a_max = 2.0  # m/s^2

        # Logs of adaptive values
        self.last_horizon = self.N
        self.last_q_scale = 1.0
        self.last_r_scale = 1.0

        # Upgrade 1 & 2 states
        self.current_horizon = float(self.N)
        self.last_target_horizon = self.N
        self.last_actual_horizon = self.N
        self.consecutive_cycles = 0
        self.switch_count = 0
        self.step_counter = 0

        self.prev_vel_cmd = np.zeros(3)
        self.vel_filter_coef = 0.20  # configurable coefficient

        self.last_raw_vel_cmd = np.zeros(3)
        self.last_filtered_vel_cmd = np.zeros(3)
        self.last_target_horizon_logged = self.N
        self.last_actual_horizon_logged = self.N
        self.switching_frequency = 0.0

    def compute_control(
        self,
        cur_pos: np.ndarray,
        cur_vel: np.ndarray,
        target_wp: np.ndarray,
        obstacle_density: float = 0.0,
        target_distance: float = 0.0,
        velocity_magnitude: float = 0.0,
        waypoint_curvature: float = 0.0,
        adaptive: bool = True
    ) -> np.ndarray:
        """Computes the optimal desired velocity vector toward a target waypoint.

        Parameters
        ----------
        cur_pos : np.ndarray
            (3,)-shaped array of the current drone 3D position.
        cur_vel : np.ndarray
            (3,)-shaped array of the current drone 3D velocity vector.
        target_wp : np.ndarray
            (3,)-shaped array of the target waypoint coordinates.
        obstacle_density : float
            Fraction of local space occupied by obstacles.
        target_distance : float
            Distance to the final target sphere.
        velocity_magnitude : float
            Current drone speed.
        waypoint_curvature : float
            Curvature of the path waypoints.
        adaptive : bool
            Whether to use adaptive horizons and cost weight scaling.

        Returns
        -------
        np.ndarray
            (3,)-shaped array of desired velocities [vx, vy, vz].
        """
        # 1. Select Horizon (N) based on complexity
        self.step_counter += 1
        if not adaptive:
            N = self.N
            q_pos_scale = 1.0
            r_scale = 1.0
            self.last_target_horizon_logged = self.N
            self.last_actual_horizon_logged = self.N
        else:
            if obstacle_density < 0.05:
                target_horizon = 10
            elif obstacle_density < 0.20:
                target_horizon = 20
            else:
                target_horizon = 30

            self.last_target_horizon_logged = target_horizon

            # Only switch if target horizon is stable for >= 5 consecutive control cycles
            if target_horizon == self.last_target_horizon:
                self.consecutive_cycles += 1
            else:
                self.consecutive_cycles = 1
                self.last_target_horizon = target_horizon

            if self.consecutive_cycles >= 5:
                self.current_horizon = 0.9 * self.current_horizon + 0.1 * target_horizon

            N = min([10, 20, 30], key=lambda x: abs(x - self.current_horizon))

            if N != self.last_actual_horizon:
                self.switch_count += 1
                self.last_actual_horizon = N

            self.last_actual_horizon_logged = N
            self.switching_frequency = self.switch_count / self.step_counter

            # 2. Adaptive Cost Weights
            # Increase tracking weight near obstacles (scale Q position weights)
            q_pos_scale = 1.0 + 4.0 * min(1.0, obstacle_density / 0.3)
            
            # Increase smoothness weight at high velocity (scale R control weights)
            r_scale = 1.0 + 3.0 * min(1.0, velocity_magnitude / 0.15)

        self.last_horizon = N
        self.last_q_scale = q_pos_scale
        self.last_r_scale = r_scale

        # Retrieve precomputed matrices
        M = self.M_dict[N]
        C = self.C_dict[N]

        # Construct dynamic Q_d and R_d matrices
        q_single = np.diag([12.0 * q_pos_scale, 12.0 * q_pos_scale, 12.0 * q_pos_scale, 0.4, 0.4, 0.4])
        Q_d = np.kron(np.eye(N), q_single)

        r_single = np.diag([1.2 * r_scale, 1.2 * r_scale, 1.2 * r_scale])
        R_d = np.kron(np.eye(N), r_single)

        # Solve for the unconstrained optimal gain matrix K_mpc
        H_cost = C.T @ Q_d @ C + R_d
        K_mpc = np.linalg.inv(H_cost) @ C.T @ Q_d

        # Current initial state z0
        z0 = np.concatenate([cur_pos, cur_vel])

        # Reference trajectory Z_ref: hold position at waypoint with zero velocity
        z_ref_single = np.concatenate([target_wp, np.zeros(3)])
        Z_ref = np.kron(np.ones(N), z_ref_single)

        # Solve unconstrained MPC trajectory
        error_term = M @ z0 - Z_ref
        U = -K_mpc @ error_term

        # Extract the first control input (acceleration for current step)
        u0 = U[0:3]

        # Enforce acceleration limit
        u0_clipped = np.clip(u0, -self.a_max, self.a_max)

        # Desired velocity command
        vel_cmd = cur_vel + u0_clipped * self.dt

        # Enforce velocity limits
        vel_cmd[0] = np.clip(vel_cmd[0], -self.vxy_max, self.vxy_max)
        vel_cmd[1] = np.clip(vel_cmd[1], -self.vxy_max, self.vxy_max)
        vel_cmd[2] = np.clip(vel_cmd[2], -self.vz_max, self.vz_max)

        # Apply low-pass filter
        self.last_raw_vel_cmd = np.copy(vel_cmd)
        vel_cmd_filtered = (1.0 - self.vel_filter_coef) * self.prev_vel_cmd + self.vel_filter_coef * vel_cmd
        self.prev_vel_cmd = np.copy(vel_cmd_filtered)
        self.last_filtered_vel_cmd = np.copy(vel_cmd_filtered)

        return vel_cmd_filtered
