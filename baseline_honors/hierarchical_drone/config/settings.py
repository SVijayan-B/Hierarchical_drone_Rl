from dataclasses import dataclass


@dataclass
class SimConfig:
    pyb_freq: int = 240
    ctrl_freq: int = 120
    rl_freq: int = 10
    episode_sec: int = 60
    gui: bool = False
    obstacles: bool = False
    # Wind disturbance amplitude in Newtons along world axes [x, y, z].
    wind_disturbance: tuple = (0.0, 0.0, 0.0)
    # Wind oscillation frequencies (Hz) for x, y, z components.
    wind_freq_hz: tuple = (0.35, 0.23, 0.17)
    # Random gust factor [0..1], scaled by disturbance amplitude.
    wind_gust_scale: float = 0.12


@dataclass
class DroneConfig:
    mass_kg: float = 0.027
    arm_m: float = 0.0397
    kf: float = 3.16e-10
    km: float = 7.94e-12
    min_rpm: float = 0.0
    max_rpm: float = 22000.0
    hover_rpm: float = 14500.0


@dataclass
class SensorConfig:
    ultrasonic_max_range: float = 3.0
    ultrasonic_noise_std: float = 0.01
    imu_rate_noise_std: float = 0.002
    imu_angle_noise_std: float = 0.001
    imu_acc_noise_std: float = 0.03


@dataclass
class TaskConfig:
    world_xy_limit: float = 2.5
    world_z_min: float = 0.1
    world_z_max: float = 2.2
    target_threshold_m: float = 0.60
    obstacle_count: int = 0
    obstacle_min_size: float = 0.10
    obstacle_max_size: float = 0.35


@dataclass
class ActionConfig:
    vxy_max: float = 0.08
    vz_max: float = 0.08
    yaw_rate_max: float = 1.2
    action_smoothing_alpha: float = 0.04
    action_rate_limit: float = 0.03


@dataclass
class PIDGains:
    kp: float
    ki: float
    kd: float
    integrator_limit: float
    output_limit: float


VELOCITY_XY_GAINS = PIDGains(kp=3.2, ki=0.25, kd=0.35, integrator_limit=0.8, output_limit=0.45)
VELOCITY_Z_GAINS = PIDGains(kp=4.0, ki=0.45, kd=0.45, integrator_limit=0.8, output_limit=8.0)
ATT_ROLL_GAINS = PIDGains(kp=6.0, ki=0.05, kd=0.25, integrator_limit=0.2, output_limit=1.8)
ATT_PITCH_GAINS = PIDGains(kp=6.0, ki=0.05, kd=0.25, integrator_limit=0.2, output_limit=1.8)
ATT_YAW_GAINS = PIDGains(kp=3.6, ki=0.05, kd=0.18, integrator_limit=0.3, output_limit=1.2)
RATE_ROLL_GAINS = PIDGains(kp=0.11, ki=0.03, kd=0.002, integrator_limit=0.8, output_limit=0.9)
RATE_PITCH_GAINS = PIDGains(kp=0.11, ki=0.03, kd=0.002, integrator_limit=0.8, output_limit=0.9)
RATE_YAW_GAINS = PIDGains(kp=0.16, ki=0.05, kd=0.002, integrator_limit=0.8, output_limit=0.7)
