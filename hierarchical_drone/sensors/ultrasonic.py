import numpy as np
import pybullet as p


class UltrasonicArray:
    """5-beam ultrasonic sensor (front, left, right, rear, down)."""

    def __init__(self, max_range=3.0, noise_std=0.01):
        self.max_range = max_range
        self.noise_std = noise_std

    def read(self, drone_id, pos, rpy, client_id, ignore_ids=None):
        if ignore_ids is None:
            ignore_ids = []
        yaw = rpy[2]
        dirs_body = {
            "front": np.array([1.0, 0.0, 0.0]),
            "left": np.array([0.0, 1.0, 0.0]),
            "right": np.array([0.0, -1.0, 0.0]),
            "rear": np.array([-1.0, 0.0, 0.0]),
            "down": np.array([0.0, 0.0, -1.0]),
        }

        c, s = np.cos(yaw), np.sin(yaw)
        rot_z = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

        offset_dist = 0.06
        out = {}
        for name, d in dirs_body.items():
            d_world = rot_z @ d if name != "down" else d
            # Offset start to prevent self-collisions with the drone body
            start = pos + np.array([0.0, 0.0, 0.03]) + d_world * offset_dist
            end = start + d_world * (self.max_range - offset_dist)
            
            hit = p.rayTest(start.tolist(), end.tolist(), physicsClientId=client_id)[0]
            if hit[0] != -1 and hit[0] != drone_id and hit[0] not in ignore_ids:
                frac = float(hit[2])
                dist = offset_dist + frac * (self.max_range - offset_dist)
            else:
                dist = self.max_range
                
            dist += float(np.random.normal(0.0, self.noise_std))
            out[name] = float(np.clip(dist, 0.0, self.max_range))

        return out
