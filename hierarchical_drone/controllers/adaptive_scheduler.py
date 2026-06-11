import numpy as np


class AdaptiveGainScheduler:
    """Computes an adaptive PID gain scale based on drone observation features."""

    def __init__(self):
        pass

    def get_gain_scale(
        self,
        velocity_mag: float,
        angular_rate_mag: float,
        distance_to_target: float,
        wind_disturbance_mag: float,
    ) -> float:
        """Computes the gain scale based on observations.

        Returns a float between 0.5 and 1.5.

        Parameters
        ----------
        velocity_mag : float
            Magnitude of drone's linear velocity vector.
        angular_rate_mag : float
            Magnitude of drone's angular velocity vector.
        distance_to_target : float
            Distance to target.
        wind_disturbance_mag : float
            Magnitude of active wind disturbance force.

        Returns
        -------
        float
            The PID gain scaling factor between 0.5 and 1.5.

        """
        # Heuristics:
        # 1. Wind disturbance: Increase gains to resist displacement.
        #    Scale up to +0.3 based on wind magnitude (normalized around 2.0 N max force).
        wind_effect = 0.3 * min(1.0, wind_disturbance_mag / 2.0)

        # 2. Distance to target: Stiffen controls when far to track fast;
        #    soften controls when very close to target to avoid overshoot.
        #    Scale up to +0.2 based on distance (normalized around 2.5m).
        dist_effect = 0.2 * min(1.0, distance_to_target / 2.5)

        # 3. Velocity magnitude: Reduce gains when moving very fast to avoid overshooting
        #    and damp oscillations.
        #    Scale down to -0.2 (normalized around 1.5 m/s).
        vel_effect = -0.2 * min(1.0, velocity_mag / 1.5)

        # 4. Angular rate magnitude: Reduce gains at high rotation rates to prevent saturation/instability.
        #    Scale down to -0.3 (normalized around 5.0 rad/s).
        ang_effect = -0.3 * min(1.0, angular_rate_mag / 5.0)

        gain_scale = 1.0 + wind_effect + dist_effect + vel_effect + ang_effect
        # Output a gain_scale between 0.5 and 1.5
        gain_scale = float(np.clip(gain_scale, 0.5, 1.5))
        return gain_scale
