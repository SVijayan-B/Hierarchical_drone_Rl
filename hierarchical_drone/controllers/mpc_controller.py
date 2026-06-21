import numpy as np


class MPCController:
    """Model Predictive Controller for 3D trajectory tracking with velocity and acceleration constraints."""

    def __init__(self, horizon: int = 20, dt: float = 0.1):
        self.N = horizon
        self.dt = dt

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

        # Build prediction matrices M and C such that:
        # Z = M * z0 + C * U
        # where Z = [z_1^T, ..., z_N^T]^T  (6N x 1)
        #       U = [u_0^T, ..., u_{N-1}^T]^T  (3N x 1)
        self.M = []
        for i in range(1, self.N + 1):
            self.M.append(np.linalg.matrix_power(self.A, i))
        self.M = np.vstack(self.M)  # Shape (6N, 6)

        self.C = np.zeros((6 * self.N, 3 * self.N))
        for r in range(self.N):
            for c in range(r + 1):
                power = r - c
                self.C[r * 6 : (r + 1) * 6, c * 3 : (c + 1) * 3] = np.linalg.matrix_power(self.A, power) @ self.B

        # Weight matrices for optimization cost:
        # Q tracks position error and drift (6N x 6N)
        # R penalizes control inputs/accelerations (3N x 3N)
        q_single = np.diag([12.0, 12.0, 12.0, 0.4, 0.4, 0.4])
        self.Q_d = np.kron(np.eye(self.N), q_single)

        r_single = np.diag([1.2, 1.2, 1.2])
        self.R_d = np.kron(np.eye(self.N), r_single)

        # Precompute unconstrained optimal gain matrix:
        # H = 2 * (C^T * Q_d * C + R_d)
        # g = 2 * C^T * Q_d * (M * z0 - Z_ref)
        # U* = -H^-1 * g = -K_mpc * (M * z0 - Z_ref)
        # where K_mpc = (C^T * Q_d * C + R_d)^-1 * C^T * Q_d
        self.H_cost = self.C.T @ self.Q_d @ self.C + self.R_d
        self.K_mpc = np.linalg.inv(self.H_cost) @ self.C.T @ self.Q_d

        # Performance constraints
        self.vxy_max = 0.15
        self.vz_max = 0.15
        self.a_max = 2.0  # m/s^2

    def compute_control(self, cur_pos: np.ndarray, cur_vel: np.ndarray, target_wp: np.ndarray) -> np.ndarray:
        """Computes the optimal desired velocity vector toward a target waypoint.

        Parameters
        ----------
        cur_pos : np.ndarray
            (3,)-shaped array of the current drone 3D position.
        cur_vel : np.ndarray
            (3,)-shaped array of the current drone 3D velocity vector.
        target_wp : np.ndarray
            (3,)-shaped array of the target waypoint coordinates.

        Returns
        -------
        np.ndarray
            (3,)-shaped array of desired velocities [vx, vy, vz].

        """
        # Current initial state z0
        z0 = np.concatenate([cur_pos, cur_vel])

        # Reference trajectory Z_ref: hold position at waypoint with zero velocity
        z_ref_single = np.concatenate([target_wp, np.zeros(3)])
        Z_ref = np.kron(np.ones(self.N), z_ref_single)

        # Solve unconstrained MPC trajectory
        error_term = self.M @ z0 - Z_ref
        U = -self.K_mpc @ error_term

        # Extract the first control input (acceleration for current step)
        u0 = U[0:3]

        # Enforce acceleration limit
        u0_clipped = np.clip(u0, -self.a_max, self.a_max)

        # Desired velocity vector computed from dynamics
        vel_cmd = cur_vel + u0_clipped * self.dt

        # Enforce velocity limits
        vel_cmd[0] = np.clip(vel_cmd[0], -self.vxy_max, self.vxy_max)
        vel_cmd[1] = np.clip(vel_cmd[1], -self.vxy_max, self.vxy_max)
        vel_cmd[2] = np.clip(vel_cmd[2], -self.vz_max, self.vz_max)

        return vel_cmd
