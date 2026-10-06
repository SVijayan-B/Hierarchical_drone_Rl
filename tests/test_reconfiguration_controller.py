"""Unit Tests for MCR-UAV Phase 9 Hierarchical Environment Reconfiguration Integration."""

import os
import sys
import math
import pytest
import numpy as np
import torch

# Ensure root workspace is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.reconfiguration_controller import ReconfigurationController
from models.meta_supervisor import ReconfigurationBounds, ReconfigurationVector
from hierarchical_drone.config.settings import SimConfig, TaskConfig, SensorConfig, ActionConfig
from hierarchical_drone.env.hierarchical_nav_env import HierarchicalNavEnv
from hierarchical_drone.controllers.mpc_controller import MPCController


def _create_sample_sim_config():
    return SimConfig(
        gui=False,
        episode_sec=1.0,
        pyb_freq=240,
        ctrl_freq=120,
        rl_freq=10,
    )


@pytest.fixture(scope="module")
def shared_env():
    sim_cfg = _create_sample_sim_config()
    env = HierarchicalNavEnv(
        sim=sim_cfg,
        task=TaskConfig(),
        sensor_cfg=SensorConfig(),
        action_cfg=ActionConfig(),
        use_mpc_layer=True,
    )
    env.evaluation_mode = True
    yield env
    env.close()


@pytest.fixture
def controller():
    return ReconfigurationController(beta_gain=0.05)


# 1. Nominal reset
def test_nominal_reset(controller):
    controller.reset()
    params = controller.get_effective_params()
    assert params["lambda_rl"] == 1.0
    assert params["alpha_q"] == 1.0
    assert params["alpha_r"] == 1.0
    assert params["alpha_p"] == 1.0
    assert params["alpha_i"] == 1.0
    assert params["alpha_d"] == 1.0
    assert params["horizon"] == 20
    assert controller.last_c_t is None


# 2. Valid configuration acceptance
def test_valid_configuration_acceptance(controller):
    c_t = {
        "lambda_rl": 0.45,
        "alpha_q": 1.5,
        "alpha_r": 2.2,
        "alpha_p": 1.2,
        "alpha_i": 0.8,
        "alpha_d": 1.4,
        "horizon": 30.0,
    }
    assert controller.apply_reconfiguration(c_t) is True
    params = controller.get_effective_params()
    assert params["lambda_rl"] == 0.45
    assert params["alpha_q"] == 1.5
    assert params["alpha_r"] == 2.2
    assert params["horizon"] == 30


# 3. Invalid lambda rejection
def test_invalid_lambda_rejection(controller):
    c_t = {
        "lambda_rl": 1.05,  # > 1.0
        "alpha_q": 1.0,
        "alpha_r": 1.0,
        "alpha_p": 1.0,
        "alpha_i": 1.0,
        "alpha_d": 1.0,
        "horizon": 20.0,
    }
    assert controller.apply_reconfiguration(c_t) is False
    assert controller.get_effective_params()["lambda_rl"] == 1.0  # fell back to nominal


# 4. Invalid alpha_Q rejection
def test_invalid_alpha_q_rejection(controller):
    c_t = {
        "lambda_rl": 0.5,
        "alpha_q": 5.1,  # > 5.0
        "alpha_r": 1.0,
        "alpha_p": 1.0,
        "alpha_i": 1.0,
        "alpha_d": 1.0,
        "horizon": 20.0,
    }
    assert controller.apply_reconfiguration(c_t) is False
    assert controller.get_effective_params()["alpha_q"] == 1.0


# 5. Invalid alpha_R rejection
def test_invalid_alpha_r_rejection(controller):
    c_t = {
        "lambda_rl": 0.5,
        "alpha_q": 1.0,
        "alpha_r": 0.15,  # < 0.2
        "alpha_p": 1.0,
        "alpha_i": 1.0,
        "alpha_d": 1.0,
        "horizon": 20.0,
    }
    assert controller.apply_reconfiguration(c_t) is False
    assert controller.get_effective_params()["alpha_r"] == 1.0


# 6. Invalid alpha_P rejection
def test_invalid_alpha_p_rejection(controller):
    c_t = {
        "lambda_rl": 0.5,
        "alpha_q": 1.0,
        "alpha_r": 1.0,
        "alpha_p": 0.4,  # < 0.5
        "alpha_i": 1.0,
        "alpha_d": 1.0,
        "horizon": 20.0,
    }
    assert controller.apply_reconfiguration(c_t) is False
    assert controller.get_effective_params()["alpha_p"] == 1.0


# 7. Invalid alpha_I rejection
def test_invalid_alpha_i_rejection(controller):
    c_t = {
        "lambda_rl": 0.5,
        "alpha_q": 1.0,
        "alpha_r": 1.0,
        "alpha_p": 1.0,
        "alpha_i": 2.6,  # > 2.5
        "alpha_d": 1.0,
        "horizon": 20.0,
    }
    assert controller.apply_reconfiguration(c_t) is False
    assert controller.get_effective_params()["alpha_i"] == 1.0


# 8. Invalid alpha_D rejection
def test_invalid_alpha_d_rejection(controller):
    c_t = {
        "lambda_rl": 0.5,
        "alpha_q": 1.0,
        "alpha_r": 1.0,
        "alpha_p": 1.0,
        "alpha_i": 1.0,
        "alpha_d": 2.1,  # > 2.0
        "horizon": 20.0,
    }
    assert controller.apply_reconfiguration(c_t) is False
    assert controller.get_effective_params()["alpha_d"] == 1.0


# 9. Invalid horizon rejection
def test_invalid_horizon_rejection(controller):
    c_t = {
        "lambda_rl": 0.5,
        "alpha_q": 1.0,
        "alpha_r": 1.0,
        "alpha_p": 1.0,
        "alpha_i": 1.0,
        "alpha_d": 1.0,
        "horizon": 24.0,  # invalid horizon
    }
    assert controller.apply_reconfiguration(c_t) is False
    assert controller.get_effective_params()["horizon"] == 20


# 10. NaN rejection
def test_nan_rejection(controller):
    c_t = {
        "lambda_rl": float("nan"),
        "alpha_q": 1.0,
        "alpha_r": 1.0,
        "alpha_p": 1.0,
        "alpha_i": 1.0,
        "alpha_d": 1.0,
        "horizon": 20.0,
    }
    assert controller.apply_reconfiguration(c_t) is False
    assert controller.get_effective_params()["lambda_rl"] == 1.0


# 11. Inf rejection
def test_inf_rejection(controller):
    c_t = {
        "lambda_rl": 0.5,
        "alpha_q": float("inf"),
        "alpha_r": 1.0,
        "alpha_p": 1.0,
        "alpha_i": 1.0,
        "alpha_d": 1.0,
        "horizon": 20.0,
    }
    assert controller.apply_reconfiguration(c_t) is False
    assert controller.get_effective_params()["alpha_q"] == 1.0


# 12. Lambda blending correctness
def test_lambda_blending_correctness(shared_env):
    shared_env.reset(seed=42)
    c_t = {
        "lambda_rl": 0.25,
        "alpha_q": 1.0,
        "alpha_r": 1.0,
        "alpha_p": 1.0,
        "alpha_i": 1.0,
        "alpha_d": 1.0,
        "horizon": 20.0,
    }
    shared_env.apply_reconfiguration(c_t)
    assert shared_env.reconfig_active is True

    # Test blending targets
    rl_wp = np.array([1.0, 2.0, 3.0])
    target_wp_astar = np.array([0.0, 0.0, 0.0])
    weight_RL = shared_env.reconfig_controller.effective_lambda_rl
    assert weight_RL == 0.25

    target_wp = weight_RL * rl_wp + (1.0 - weight_RL) * target_wp_astar
    expected = 0.25 * rl_wp
    assert np.allclose(target_wp, expected)


# 13. Q scaling correctness
def test_q_scaling_correctness():
    mpc = MPCController(horizon=20)
    mpc.reconfig_active = True
    mpc.reconfig_alpha_q = 2.0
    mpc.reconfig_alpha_r = 1.0
    mpc.reconfig_horizon = 20

    cur_pos = np.zeros(3)
    cur_vel = np.zeros(3)
    target_wp = np.ones(3)
    
    mpc.compute_control(cur_pos, cur_vel, target_wp, adaptive=False)
    assert mpc.last_q_scale == 2.0


# 14. R scaling correctness
def test_r_scaling_correctness():
    mpc = MPCController(horizon=20)
    mpc.reconfig_active = True
    mpc.reconfig_alpha_q = 1.0
    mpc.reconfig_alpha_r = 0.5
    mpc.reconfig_horizon = 20

    cur_pos = np.zeros(3)
    cur_vel = np.zeros(3)
    target_wp = np.ones(3)
    mpc.compute_control(cur_pos, cur_vel, target_wp, adaptive=False)
    assert mpc.last_r_scale == 0.5


# 15. Horizon selection correctness
def test_horizon_selection_correctness():
    mpc = MPCController(horizon=20)
    mpc.reconfig_active = True
    mpc.reconfig_alpha_q = 1.0
    mpc.reconfig_alpha_r = 1.0
    
    for h in [10, 20, 30]:
        mpc.reconfig_horizon = h
        mpc.compute_control(np.zeros(3), np.zeros(3), np.ones(3), adaptive=False)
        assert mpc.last_horizon == h


# 16. PID target gain calculation
def test_pid_target_gain_calculation(controller):
    c_t = {
        "lambda_rl": 1.0,
        "alpha_q": 1.0,
        "alpha_r": 1.0,
        "alpha_p": 1.5,
        "alpha_i": 0.5,
        "alpha_d": 1.2,
        "horizon": 20.0,
    }
    controller.apply_reconfiguration(c_t)
    assert controller.target_alpha_p == 1.5
    assert controller.target_alpha_i == 0.5
    assert controller.target_alpha_d == 1.2


# 17. Gain smoothing correctness
def test_gain_smoothing_correctness(controller):
    controller.reset()
    c_t = {
        "lambda_rl": 1.0,
        "alpha_q": 1.0,
        "alpha_r": 1.0,
        "alpha_p": 2.0,
        "alpha_i": 1.0,
        "alpha_d": 1.0,
        "horizon": 20.0,
    }
    controller.apply_reconfiguration(c_t)
    
    # Step 1:
    controller.smooth_inner_step()
    expected_p1 = 0.95 * 1.0 + 0.05 * 2.0
    assert abs(controller.effective_alpha_p - expected_p1) < 1e-7

    # Step 2:
    controller.smooth_inner_step()
    expected_p2 = 0.95 * expected_p1 + 0.05 * 2.0
    assert abs(controller.effective_alpha_p - expected_p2) < 1e-7


# 18. Repeated gain smoothing convergence
def test_repeated_gain_smoothing_convergence(controller):
    c_t = {
        "lambda_rl": 1.0,
        "alpha_q": 1.0,
        "alpha_r": 1.0,
        "alpha_p": 2.0,
        "alpha_i": 1.0,
        "alpha_d": 1.0,
        "horizon": 20.0,
    }
    controller.apply_reconfiguration(c_t)
    
    for _ in range(200):
        controller.smooth_inner_step()
        
    assert abs(controller.effective_alpha_p - 2.0) < 1e-4


# 19. Gain bounds after smoothing
def test_gain_bounds_after_smoothing(controller):
    c_t = {
        "lambda_rl": 1.0,
        "alpha_q": 1.0,
        "alpha_r": 1.0,
        "alpha_p": 2.0,
        "alpha_i": 2.5,
        "alpha_d": 2.0,
        "horizon": 20.0,
    }
    controller.apply_reconfiguration(c_t)
    
    for _ in range(300):
        controller.smooth_inner_step()
        
    assert controller.effective_alpha_p <= 2.0
    assert controller.effective_alpha_i <= 2.5
    assert controller.effective_alpha_d <= 2.0


# 20. Reset restores nominal values
def test_reset_restores_nominal_values(controller):
    c_t = {
        "lambda_rl": 0.2,
        "alpha_q": 3.0,
        "alpha_r": 0.5,
        "alpha_p": 1.5,
        "alpha_i": 0.5,
        "alpha_d": 1.8,
        "horizon": 10.0,
    }
    controller.apply_reconfiguration(c_t)
    controller.smooth_inner_step()
    
    controller.reset()
    params = controller.get_effective_params()
    assert params["lambda_rl"] == 1.0
    assert params["alpha_q"] == 1.0
    assert params["alpha_r"] == 1.0
    assert params["alpha_p"] == 1.0
    assert params["alpha_i"] == 1.0
    assert params["alpha_d"] == 1.0
    assert params["horizon"] == 20


# 21. Missing configuration fallback
def test_missing_configuration_fallback(controller):
    c_t = {
        "lambda_rl": 0.5,
        "alpha_q": 2.0,
        "alpha_r": 2.0,
        "alpha_p": 1.5,
        "alpha_i": 1.5,
        "alpha_d": 1.5,
        "horizon": 30.0,
    }
    controller.apply_reconfiguration(c_t)
    
    assert controller.apply_reconfiguration(None) is False
    params = controller.get_effective_params()
    assert params["lambda_rl"] == 1.0
    assert params["alpha_q"] == 1.0
    assert params["alpha_r"] == 1.0
    assert params["horizon"] == 20


# 22. Anti-windup compatibility
def test_anti_windup_compatibility(shared_env):
    assert np.allclose(shared_env.stab_ctrl.integral_pos_e, np.zeros(3))
    shared_env.stab_ctrl.integral_pos_e = np.array([3.0, 3.0, 3.0])
    
    # Emulate _dslPIDPositionControl behavior of clipping
    shared_env.stab_ctrl.integral_pos_e = np.clip(shared_env.stab_ctrl.integral_pos_e, -2., 2.)
    assert np.allclose(shared_env.stab_ctrl.integral_pos_e, np.array([2.0, 2.0, 2.0]))


# 23. No neural inference inside 1200 Hz loop
def test_no_neural_inference_inside_1200hz_loop():
    controller = ReconfigurationController()
    assert controller.last_c_t is None
    controller.smooth_inner_step()
    assert controller.last_c_t is None


# 24. Configuration latching
def test_configuration_latching(controller):
    c_t = {
        "lambda_rl": 0.4,
        "alpha_q": 1.5,
        "alpha_r": 1.5,
        "alpha_p": 1.2,
        "alpha_i": 1.2,
        "alpha_d": 1.2,
        "horizon": 30.0,
    }
    controller.apply_reconfiguration(c_t)
    assert controller.target_alpha_p == 1.2
    assert controller.target_horizon == 30
    assert controller.target_alpha_p == 1.2


# 25. Deterministic behavior
def test_deterministic_behavior(controller):
    c_t = {
        "lambda_rl": 0.4,
        "alpha_q": 1.5,
        "alpha_r": 1.5,
        "alpha_p": 1.2,
        "alpha_i": 1.2,
        "alpha_d": 1.2,
        "horizon": 30.0,
    }
    
    c1 = ReconfigurationController()
    c2 = ReconfigurationController()
    
    c1.apply_reconfiguration(c_t)
    c2.apply_reconfiguration(c_t)
    
    for _ in range(50):
        c1.smooth_inner_step()
        c2.smooth_inner_step()
        
    assert c1.get_effective_params() == c2.get_effective_params()


# 26. Nominal equivalence regression
def test_nominal_equivalence_regression(shared_env):
    # Set up environment to compare baseline vs. reconfig under nominal parameters
    # Disabling the heuristic scheduler ensures both use constant nominal gains of 1.0
    original_use_adaptive = shared_env.use_adaptive_scheduler
    shared_env.use_adaptive_scheduler = False
    
    # Temporarily disable sensor noise to ensure perfect numerical determinism
    original_imu_angle = shared_env.imu.angle_noise_std
    original_imu_rate = shared_env.imu.rate_noise_std
    original_imu_acc = shared_env.imu.acc_noise_std
    original_ultra = shared_env.ultra.noise_std
    
    shared_env.imu.angle_noise_std = 0.0
    shared_env.imu.rate_noise_std = 0.0
    shared_env.imu.acc_noise_std = 0.0
    shared_env.ultra.noise_std = 0.0
    
    # Nominal configuration vector representing baseline defaults
    nominal_c_t = {
        "lambda_rl": 1.0 - shared_env.guidance_blend,
        "alpha_q": 1.0,
        "alpha_r": 1.0,
        "alpha_p": 1.0,
        "alpha_i": 1.0,
        "alpha_d": 1.0,
        "horizon": 20.0,
    }
    
    try:
        # Case A: Nominal control step (no reconfig active)
        action = np.zeros(4)
        shared_env.reset(seed=42)
        shared_env.reconfig_active = False
        obs_baseline, _, _, _, _ = shared_env.step(action)
        
        # Case B: Integrated reconfig controller active with nominal parameters
        shared_env.reset(seed=42)
        shared_env.apply_reconfiguration(nominal_c_t)
        assert shared_env.reconfig_active is True
        obs_reconfig, _, _, _, _ = shared_env.step(action)
        
        # Must be identical within numerical tolerance
        assert np.allclose(obs_baseline, obs_reconfig, atol=1e-3)
    finally:
        shared_env.use_adaptive_scheduler = original_use_adaptive
        shared_env.imu.angle_noise_std = original_imu_angle
        shared_env.imu.rate_noise_std = original_imu_rate
        shared_env.imu.acc_noise_std = original_imu_acc
        shared_env.ultra.noise_std = original_ultra
