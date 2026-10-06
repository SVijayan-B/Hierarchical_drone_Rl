import os
import csv
import time
import numpy as np

class TelemetryLogger:
    def __init__(self, save_dir="results/telemetry"):
        self.save_dir = save_dir
        self.buffer = []
        self.run_name = "default"
        self.config_name = "default"
        self.round_idx = 1

    def set_run_info(self, run_name: str, config_name: str, round_idx: int):
        self.run_name = run_name
        self.config_name = config_name
        self.round_idx = round_idx

    def reset(self):
        self.buffer = []

    def log_step(self, env):
        # Extract states and details safely
        s = env.last_state
        
        # Get true orientation and angular velocity
        rpy_true = s[7:10]
        rates_true = s[13:16]
        
        # Get measured IMU readings if available, fallback to true values
        imu = getattr(env, "last_imu_readings", None)
        if imu is None:
            imu = {
                "roll": float(rpy_true[0]),
                "pitch": float(rpy_true[1]),
                "yaw": float(rpy_true[2]),
                "p": float(rates_true[0]),
                "q": float(rates_true[1]),
                "r": float(rates_true[2]),
                "ax": 0.0,
                "ay": 0.0,
                "az": 0.0,
            }
            
        # Get measured ultrasonic readings if available
        ultra = getattr(env, "last_ultra_readings", None)
        if ultra is None:
            ultra = {"front": 3.0, "left": 3.0, "right": 3.0, "rear": 3.0, "down": 3.0}

        # Calculate pos_error and target_error
        pos_error = float(np.linalg.norm(s[0:3] - env.pos_ref))
        target_error = float(np.linalg.norm(s[0:3] - env.target))
        
        # Applied motor action (RPMs)
        applied_rpm = getattr(env, "last_applied_rpm", None)
        if applied_rpm is None:
            applied_rpm = np.zeros(4)

        # Wind disturbance vector
        wind_vec = getattr(env, "active_wind_disturbance", np.zeros(3))
        
        # Motor efficiencies
        motor_eff = getattr(env, "motor_efficiencies", np.ones(4))

        step_record = {
            "step": int(env.step_count),
            "control_step": int(env.control_step_counter),
            "time": float(env.step_count * env.ctrl_dt * env.rl_every_n),
            # True States
            "pos_x": float(s[0]),
            "pos_y": float(s[1]),
            "pos_z": float(s[2]),
            "vel_x": float(s[10]),
            "vel_y": float(s[11]),
            "vel_z": float(s[12]),
            "roll": float(rpy_true[0]),
            "pitch": float(rpy_true[1]),
            "yaw": float(rpy_true[2]),
            "p": float(rates_true[0]),
            "q": float(rates_true[1]),
            "r": float(rates_true[2]),
            "acc_x": float(env.prev_acc[0]),
            "acc_y": float(env.prev_acc[1]),
            "acc_z": float(env.prev_acc[2]),
            # Measured IMU States
            "imu_roll": float(imu.get("roll", 0.0)),
            "imu_pitch": float(imu.get("pitch", 0.0)),
            "imu_yaw": float(imu.get("yaw", 0.0)),
            "imu_p": float(imu.get("p", 0.0)),
            "imu_q": float(imu.get("q", 0.0)),
            "imu_r": float(imu.get("r", 0.0)),
            "imu_acc_x": float(imu.get("ax", 0.0)),
            "imu_acc_y": float(imu.get("ay", 0.0)),
            "imu_acc_z": float(imu.get("az", 0.0)),
            # References
            "ref_x": float(env.pos_ref[0]),
            "ref_y": float(env.pos_ref[1]),
            "ref_z": float(env.pos_ref[2]),
            "target_x": float(env.target[0]),
            "target_y": float(env.target[1]),
            "target_z": float(env.target[2]),
            "pos_error": pos_error,
            "target_error": target_error,
            # Gains
            "gain_scale_pos": float(env.gain_scale_pos),
            "gain_scale_att": float(env.gain_scale_att),
            "kp_pos_x": float(env.stab_ctrl.P_COEFF_FOR[0]),
            "kp_pos_y": float(env.stab_ctrl.P_COEFF_FOR[1]),
            "kp_pos_z": float(env.stab_ctrl.P_COEFF_FOR[2]),
            "ki_pos_x": float(env.stab_ctrl.I_COEFF_FOR[0]),
            "ki_pos_y": float(env.stab_ctrl.I_COEFF_FOR[1]),
            "ki_pos_z": float(env.stab_ctrl.I_COEFF_FOR[2]),
            "kd_pos_x": float(env.stab_ctrl.D_COEFF_FOR[0]),
            "kd_pos_y": float(env.stab_ctrl.D_COEFF_FOR[1]),
            "kd_pos_z": float(env.stab_ctrl.D_COEFF_FOR[2]),
            "kp_att_r": float(env.stab_ctrl.P_COEFF_TOR[0]),
            "kp_att_p": float(env.stab_ctrl.P_COEFF_TOR[1]),
            "kp_att_y": float(env.stab_ctrl.P_COEFF_TOR[2]),
            "ki_att_r": float(env.stab_ctrl.I_COEFF_TOR[0]),
            "ki_att_p": float(env.stab_ctrl.I_COEFF_TOR[1]),
            "ki_att_y": float(env.stab_ctrl.I_COEFF_TOR[2]),
            "kd_att_r": float(env.stab_ctrl.D_COEFF_TOR[0]),
            "kd_att_p": float(env.stab_ctrl.D_COEFF_TOR[1]),
            "kd_att_y": float(env.stab_ctrl.D_COEFF_TOR[2]),
            # MPC variables
            "mpc_horizon": float(env.mpc.last_horizon) if env.use_mpc_layer else 0.0,
            "mpc_q_scale": float(env.mpc.last_q_scale) if env.use_mpc_layer else 1.0,
            "mpc_r_scale": float(env.mpc.last_r_scale) if env.use_mpc_layer else 1.0,
            # Environment / Disturbances
            "wind_x": float(wind_vec[0]),
            "wind_y": float(wind_vec[1]),
            "wind_z": float(wind_vec[2]),
            "wind_magnitude": float(env.current_wind_magnitude),
            "motor_efficiency_1": float(motor_eff[0]),
            "motor_efficiency_2": float(motor_eff[1]),
            "motor_efficiency_3": float(motor_eff[2]),
            "motor_efficiency_4": float(motor_eff[3]),
            # Sensors
            "ultra_front": float(ultra.get("front", 3.0)),
            "ultra_left": float(ultra.get("left", 3.0)),
            "ultra_right": float(ultra.get("right", 3.0)),
            "ultra_rear": float(ultra.get("rear", 3.0)),
            "ultra_down": float(ultra.get("down", 3.0)),
            # Motor Actions (RPMs)
            "motor_rpm_1": float(applied_rpm[0]),
            "motor_rpm_2": float(applied_rpm[1]),
            "motor_rpm_3": float(applied_rpm[2]),
            "motor_rpm_4": float(applied_rpm[3]),
            # Performance metrics
            "jerk_estimate": float(env.jerk_estimate),
            "energy_step": float(env.step_energy),
            "total_energy": float(env.total_energy_consumed),
            "step_reward": float(getattr(env, "last_reward", 0.0)),
            "cumulative_reward": float(env.ep_reward),
        }
        self.buffer.append(step_record)

    def save_episode(self, success: bool):
        if not self.buffer:
            return
            
        config_dir = os.path.join(self.save_dir, "simulation_data")
        os.makedirs(config_dir, exist_ok=True)
        
        status_str = "success" if success else "failed"
        base_name = f"{self.config_name}_round_{self.round_idx}_{status_str}"
        
        # 1. Save CSV
        csv_path = os.path.join(config_dir, f"{base_name}.csv")
        headers = list(self.buffer[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(self.buffer)
            
        # 2. Save NPZ (efficient compressed numpy archive)
        npz_path = os.path.join(config_dir, f"{base_name}.npz")
        npz_data = {}
        for key in headers:
            npz_data[key] = np.array([step[key] for step in self.buffer])
        np.savez_compressed(npz_path, **npz_data)
        print(f"[TelemetryLogger] Exported run '{self.config_name}' round {self.round_idx} ({status_str}) with {len(self.buffer)} steps to CSV/NPZ.")
        self.buffer = []
