import numpy as np


class PIDController:
    """Generic PID with anti-windup, derivative damping, and output saturation."""

    def __init__(self, kp, ki, kd, integrator_limit, output_limit):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integrator_limit = integrator_limit
        self.output_limit = output_limit
        self.integrator = 0.0
        self.prev_error = 0.0
        self.prev_derivative = 0.0
        self.derivative_alpha = 0.25

    def reset(self):
        self.integrator = 0.0
        self.prev_error = 0.0
        self.prev_derivative = 0.0

    def update(self, error, dt):
        if dt <= 0.0:
            return 0.0

        p = self.kp * error

        self.integrator += error * dt
        self.integrator = float(np.clip(self.integrator, -self.integrator_limit, self.integrator_limit))
        i = self.ki * self.integrator

        deriv = (error - self.prev_error) / dt
        deriv = (1.0 - self.derivative_alpha) * self.prev_derivative + self.derivative_alpha * deriv
        d = self.kd * deriv

        out = p + i + d
        out_clamped = float(np.clip(out, -self.output_limit, self.output_limit))

        if abs(out) > self.output_limit and self.ki > 1e-9:
            self.integrator -= error * dt * 0.25
            self.integrator = float(np.clip(self.integrator, -self.integrator_limit, self.integrator_limit))

        self.prev_error = error
        self.prev_derivative = deriv
        return out_clamped
