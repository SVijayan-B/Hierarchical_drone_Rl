import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl


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


class UpgradedDSLPIDControl(DSLPIDControl):
    """Subclass of gym_pybullet_drones DSLPIDControl with Phase 6 robustness improvements."""

    def __init__(self, drone_model, g=9.8):
        # DSLPIDControl is imported at file level or inside subclass if needed.
        # Let's import it here to ensure it's loaded.
        from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
        super().__init__(drone_model=drone_model, g=g)
        self.rpy_rates_e_filtered = np.zeros(3)
        self.freeze_integrator = False
        self.deriv_alpha = 0.04

    def reset(self):
        super().reset()
        self.rpy_rates_e_filtered = np.zeros(3)
        self.freeze_integrator = False

    def _dslPIDPositionControl(self,
                               control_timestep,
                               cur_pos,
                               cur_quat,
                               cur_vel,
                               target_pos,
                               target_rpy,
                               target_vel
                               ):
        import math
        cur_rotation = np.array(p.getMatrixFromQuaternion(cur_quat)).reshape(3, 3)
        pos_e = target_pos - cur_pos
        vel_e = target_vel - cur_vel
        
        # Integrator reset after severe position disturbance (e.g. error > 0.6 m)
        pos_error_mag = np.linalg.norm(pos_e)
        if pos_error_mag > 0.6:
            self.integral_pos_e = np.zeros(3)
            
        # Anti-windup freezing
        if not self.freeze_integrator:
            self.integral_pos_e = self.integral_pos_e + pos_e * control_timestep
            self.integral_pos_e = np.clip(self.integral_pos_e, -2., 2.)
            self.integral_pos_e[2] = np.clip(self.integral_pos_e[2], -0.15, .15)
            
        #### PID target thrust #####################################
        target_thrust = np.multiply(self.P_COEFF_FOR, pos_e) \
                        + np.multiply(self.I_COEFF_FOR, self.integral_pos_e) \
                        + np.multiply(self.D_COEFF_FOR, vel_e) + np.array([0, 0, self.GRAVITY])
        
        scalar_thrust = max(0., np.dot(target_thrust, cur_rotation[:,2]))
        thrust = (math.sqrt(scalar_thrust / (4*self.KF)) - self.PWM2RPM_CONST) / self.PWM2RPM_SCALE
        
        target_z_ax = target_thrust / np.linalg.norm(target_thrust)
        target_x_c = np.array([math.cos(target_rpy[2]), math.sin(target_rpy[2]), 0])
        target_y_ax = np.cross(target_z_ax, target_x_c) / np.linalg.norm(np.cross(target_z_ax, target_x_c))
        target_x_ax = np.cross(target_y_ax, target_z_ax)
        target_rotation = (np.vstack([target_x_ax, target_y_ax, target_z_ax])).transpose()
        
        #### Target rotation #######################################
        target_euler = (Rotation.from_matrix(target_rotation)).as_euler('XYZ', degrees=False)
        
        # VELOCITY CONTROLLER UPGRADE: Limit target tilt angles adaptively based on distance
        # Keep drone level when close, allow aggressive tilt when far (from 5 degrees to 30 degrees)
        dist_to_wp = np.linalg.norm(target_pos - cur_pos)
        tilt_scale = np.clip(dist_to_wp / 1.5, 0.16, 1.0)
        max_tilt_rad = np.deg2rad(30.0) * tilt_scale
        
        target_euler[0] = np.clip(target_euler[0], -max_tilt_rad, max_tilt_rad)
        target_euler[1] = np.clip(target_euler[1], -max_tilt_rad, max_tilt_rad)
        
        return thrust, target_euler, pos_e

    def _dslPIDAttitudeControl(self,
                               control_timestep,
                               thrust,
                               cur_quat,
                               target_euler,
                               target_rpy_rates
                               ):
        cur_rotation = np.array(p.getMatrixFromQuaternion(cur_quat)).reshape(3, 3)
        cur_rpy = np.array(p.getEulerFromQuaternion(cur_quat))
        target_quat = (Rotation.from_euler('XYZ', target_euler, degrees=False)).as_quat()
        w,x,y,z = target_quat
        target_rotation = (Rotation.from_quat([w, x, y, z])).as_matrix()
        rot_matrix_e = np.dot((target_rotation.transpose()),cur_rotation) - np.dot(cur_rotation.transpose(),target_rotation)
        rot_e = np.array([rot_matrix_e[2, 1], rot_matrix_e[0, 2], rot_matrix_e[1, 0]]) 
        
        # Integrator reset after severe attitude disturbance (> 45 deg or 0.8 rad)
        att_error_mag = np.linalg.norm(rot_e)
        if att_error_mag > 0.8:
            self.integral_rpy_e = np.zeros(3)
            
        rpy_rates_e = target_rpy_rates - (cur_rpy - self.last_rpy)/control_timestep
        self.last_rpy = cur_rpy
        
        # Adaptive derivative filtering
        self.rpy_rates_e_filtered = (1.0 - self.deriv_alpha) * self.rpy_rates_e_filtered + self.deriv_alpha * rpy_rates_e
        
        # Anti-windup freezing
        if not self.freeze_integrator:
            self.integral_rpy_e = self.integral_rpy_e - rot_e*control_timestep
            self.integral_rpy_e = np.clip(self.integral_rpy_e, -1500., 1500.)
            self.integral_rpy_e[0:2] = np.clip(self.integral_rpy_e[0:2], -1., 1.)
            
        #### PID target torques ####################################
        target_torques = - np.multiply(self.P_COEFF_TOR, rot_e) \
                         + np.multiply(self.D_COEFF_TOR, self.rpy_rates_e_filtered) \
                         + np.multiply(self.I_COEFF_TOR, self.integral_rpy_e)
        target_torques = np.clip(target_torques, -3200, 3200)
        
        pwm = thrust + np.dot(self.MIXER_MATRIX, target_torques)
        
        # Check saturation to freeze integrator next step
        if np.any(pwm <= self.MIN_PWM) or np.any(pwm >= self.MAX_PWM):
            self.freeze_integrator = True
        else:
            self.freeze_integrator = False
            
        pwm = np.clip(pwm, self.MIN_PWM, self.MAX_PWM)
        return self.PWM2RPM_SCALE * pwm + self.PWM2RPM_CONST

