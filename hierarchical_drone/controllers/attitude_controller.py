import numpy as np

from hierarchical_drone.controllers.pid_controller import PIDController
from hierarchical_drone.config.settings import (
    ATT_ROLL_GAINS,
    ATT_PITCH_GAINS,
    ATT_YAW_GAINS,
    RATE_ROLL_GAINS,
    RATE_PITCH_GAINS,
    RATE_YAW_GAINS,
)


class AttitudeController:
    """Cascaded attitude->rate->torque command generator."""

    def __init__(self):
        self.roll_pid = PIDController(**ATT_ROLL_GAINS.__dict__)
        self.pitch_pid = PIDController(**ATT_PITCH_GAINS.__dict__)
        self.yaw_pid = PIDController(**ATT_YAW_GAINS.__dict__)

        self.p_rate_pid = PIDController(**RATE_ROLL_GAINS.__dict__)
        self.q_rate_pid = PIDController(**RATE_PITCH_GAINS.__dict__)
        self.r_rate_pid = PIDController(**RATE_YAW_GAINS.__dict__)

    def reset(self):
        for c in [
            self.roll_pid,
            self.pitch_pid,
            self.yaw_pid,
            self.p_rate_pid,
            self.q_rate_pid,
            self.r_rate_pid,
        ]:
            c.reset()

    def update(self, desired_rpy, desired_yaw_rate, measured_rpy, measured_rates, dt):
        roll_err = desired_rpy[0] - measured_rpy[0]
        pitch_err = desired_rpy[1] - measured_rpy[1]
        yaw_err = desired_rpy[2] - measured_rpy[2]
        yaw_err = (yaw_err + np.pi) % (2.0 * np.pi) - np.pi

        p_des = self.roll_pid.update(roll_err, dt)
        q_des = self.pitch_pid.update(pitch_err, dt)
        r_des = desired_yaw_rate + self.yaw_pid.update(yaw_err, dt)

        p_err = p_des - measured_rates[0]
        q_err = q_des - measured_rates[1]
        r_err = r_des - measured_rates[2]

        tau_x = self.p_rate_pid.update(p_err, dt)
        tau_y = self.q_rate_pid.update(q_err, dt)
        tau_z = self.r_rate_pid.update(r_err, dt)

        return tau_x, tau_y, tau_z


class MotorMixer:
    """X-quad mixer mapping thrust/torques to motor RPM."""

    def __init__(self, hover_rpm, min_rpm, max_rpm):
        self.hover_rpm = hover_rpm
        self.min_rpm = min_rpm
        self.max_rpm = max_rpm

    def mix(self, collective_thrust_cmd, tau_x, tau_y, tau_z):
        base = self.hover_rpm + collective_thrust_cmd * 1400.0
        m1 = base - tau_x + tau_y + tau_z
        m2 = base - tau_x - tau_y - tau_z
        m3 = base + tau_x - tau_y + tau_z
        m4 = base + tau_x + tau_y - tau_z
        rpm = np.array([m1, m2, m3, m4], dtype=np.float32)
        return np.clip(rpm, self.min_rpm, self.max_rpm)
