import numpy as np

from hierarchical_drone.controllers.pid_controller import PIDController
from hierarchical_drone.config.settings import VELOCITY_XY_GAINS, VELOCITY_Z_GAINS


class VelocityController:
    """Maps desired velocity to desired attitude and thrust command."""

    def __init__(self):
        self.vx_pid = PIDController(**VELOCITY_XY_GAINS.__dict__)
        self.vy_pid = PIDController(**VELOCITY_XY_GAINS.__dict__)
        self.vz_pid = PIDController(**VELOCITY_Z_GAINS.__dict__)
        self.max_tilt_rad = np.deg2rad(22.0)

    def reset(self):
        self.vx_pid.reset()
        self.vy_pid.reset()
        self.vz_pid.reset()

    def update(self, vel_des, vel_meas, yaw, dt):
        vx_err = vel_des[0] - vel_meas[0]
        vy_err = vel_des[1] - vel_meas[1]
        vz_err = vel_des[2] - vel_meas[2]

        ax_cmd = self.vx_pid.update(vx_err, dt)
        ay_cmd = self.vy_pid.update(vy_err, dt)
        thrust_delta = self.vz_pid.update(vz_err, dt)

        c, s = np.cos(yaw), np.sin(yaw)
        ax_body = c * ax_cmd + s * ay_cmd
        ay_body = -s * ax_cmd + c * ay_cmd

        pitch_des = float(np.clip(-ax_body / 9.81, -self.max_tilt_rad, self.max_tilt_rad))
        roll_des = float(np.clip(ay_body / 9.81, -self.max_tilt_rad, self.max_tilt_rad))

        return roll_des, pitch_des, thrust_delta
