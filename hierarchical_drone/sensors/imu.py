import numpy as np


class IMUSensor:
    """IMU data extraction from state vector with configurable noise."""

    def __init__(self, angle_noise_std=0.001, rate_noise_std=0.002, acc_noise_std=0.03):
        self.angle_noise_std = angle_noise_std
        self.rate_noise_std = rate_noise_std
        self.acc_noise_std = acc_noise_std
        self.prev_vel = np.zeros(3, dtype=np.float32)

    def reset(self):
        self.prev_vel[:] = 0.0

    def read(self, state, dt):
        rpy = state[7:10].copy()
        rates = state[13:16].copy()
        vel = state[10:13].copy()
        acc = (vel - self.prev_vel) / max(dt, 1e-6)
        self.prev_vel = vel

        rpy += np.random.normal(0.0, self.angle_noise_std, size=3)
        rates += np.random.normal(0.0, self.rate_noise_std, size=3)
        acc += np.random.normal(0.0, self.acc_noise_std, size=3)

        return {
            "roll": float(rpy[0]),
            "pitch": float(rpy[1]),
            "yaw": float(rpy[2]),
            "p": float(rates[0]),
            "q": float(rates[1]),
            "r": float(rates[2]),
            "ax": float(acc[0]),
            "ay": float(acc[1]),
            "az": float(acc[2]),
        }
