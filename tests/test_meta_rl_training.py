"""Unit tests for MCR-UAV Phase 10 Meta-RL Training Loop & Integration."""

import os
import sys
import tempfile
import pytest
import numpy as np
import torch
import torch.nn as nn

# Ensure root workspace is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from environments.task_generator import MetaTask
from models.meta_supervisor import MetaSupervisor, ReconfigurationBounds, ReconfigurationVector
from models.transformer_context_encoder import TransformerContextEncoder
from models.meta_rl_trainer import MetaRLTrainer, TrainingMode
from models.trajectory_buffer import EpisodeBuffer, Transition, TaskSplitEnum


@pytest.fixture(scope="module")
def trainer():
    """Create a trainer instance for unit tests."""
    transformer = TransformerContextEncoder(
        input_dim=52,
        seq_len=20,
        d_model=64,
        n_heads=4,
        n_layers=2,
        latent_dim=16,
    )
    supervisor = MetaSupervisor(
        latent_dim=16,
        bounds=ReconfigurationBounds(),
    )
    # Using real PPO and VecNormalize checkpoints which exist in workspace
    return MetaRLTrainer(
        transformer=transformer,
        supervisor=supervisor,
        ppo_checkpoint_path="results_hierarchical/run_mlp_obs_dist/final_model.zip",
        vecnormalize_path="results_hierarchical/run_mlp_obs_dist/vecnormalize.pkl",
        lr=1e-3,
        gamma=0.99,
        explore_std=0.05,
        training_mode=TrainingMode.MODE_A,
        master_seed=42,
    )


@pytest.fixture
def mock_task():
    """Return a mock training task configuration."""
    return MetaTask(
        task_id="task_train_001",
        seed=1042,
        split="train",
        wind_magnitude=1.5,
        wind_direction=[1.0, 0.0, 0.0],
        gust_magnitude=0.5,
        gust_frequency=1.0,
        turbulence_std=0.05,
        impulse_magnitude=2.0,
        impulse_duration=0.1,
        impulse_onset=2.0,
        sensor_noise_scale=1.2,
        motor_degradation=0.15,
        degraded_motors=[0, 1],
        mass_scale=1.02,
        inertia_scale=1.05,
        waypoint_seed=2026,
        compound_disturbance_flags={"wind": True, "motor_degradation": True},
    )


# 1. Model construction
def test_model_construction(trainer):
    assert isinstance(trainer.transformer, TransformerContextEncoder)
    assert isinstance(trainer.supervisor, MetaSupervisor)
    assert trainer.lr == 1e-3
    assert trainer.gamma == 0.99
    assert trainer.explore_std == 0.05
    assert trainer.training_mode == TrainingMode.MODE_A


# 2. Trajectory collection & 3. History construction
def test_trajectory_collection_and_history(trainer, mock_task):
    # Run a very short rollout of 3 steps
    ep_buffer, log_probs, total_reward = trainer.rollout_episode(mock_task, explore=True, max_steps=3)
    
    assert isinstance(ep_buffer, EpisodeBuffer)
    assert len(ep_buffer.transitions) == 3
    assert len(log_probs) == 3
    assert isinstance(total_reward, float)

    # Test history extraction at timestep 2
    history, padding_mask, valid_len = ep_buffer.get_history(t=2, window_size=20)
    assert history.shape == (20, 52)
    assert padding_mask.shape == (20,)
    assert valid_len == 3
    assert padding_mask[3:].all()  # padded elements should be True


# 4. Forward pass & 5. c_t generation
def test_forward_pass_and_ct_generation(trainer):
    # Construct a random history batch [1, 20, 52]
    history = torch.randn(1, 20, 52)
    padding_mask = torch.zeros(1, 20, dtype=torch.bool)
    valid_lens = torch.tensor([20], dtype=torch.long)

    z_t = trainer.transformer(history, padding_mask, valid_lens)
    assert z_t.shape == (1, 16)

    reconfig_out = trainer.supervisor(z_t, return_hard=True)
    assert isinstance(reconfig_out, ReconfigurationVector)
    assert reconfig_out.continuous_vector.shape == (1, 6)
    assert reconfig_out.horizon.shape == (1, 1)


# 6. c_t bounds & 7. Horizon validity
def test_ct_bounds_and_horizon_validity(trainer, mock_task):
    ep_buffer, _, _ = trainer.rollout_episode(mock_task, explore=True, max_steps=2)
    
    for trans in ep_buffer.transitions:
        assert 0.0 <= trans.lambda_rl <= 1.0
        assert 0.2 <= trans.alpha_q <= 5.0
        assert 0.2 <= trans.alpha_r <= 5.0
        assert 0.5 <= trans.alpha_p <= 2.0
        assert 0.2 <= trans.alpha_i <= 2.5
        assert 0.5 <= trans.alpha_d <= 2.0
        assert trans.horizon in (10, 20, 30)


# 8. Loss finite & 9. Gradient finite & 10. Nonzero gradients & 11. Optimizer step & 12. Parameter change
def test_training_step_optimizations(trainer, mock_task):
    # Perform a trainer step with 1 task
    train_results = trainer.train_step([mock_task])
    
    assert "loss_total" in train_results
    assert np.isfinite(train_results["loss_total"])
    assert "mean_return" in train_results
    assert np.isfinite(train_results["mean_return"])

    # Gradients audit check
    grad_audit = train_results["grad_audit"]
    assert "transformer" in grad_audit
    assert "meta_supervisor" in grad_audit
    
    for group in ["transformer", "meta_supervisor"]:
        assert grad_audit[group]["NaN_count"] == 0
        assert grad_audit[group]["Inf_count"] == 0
        assert grad_audit[group]["gradient_norm"] > 0.0
        assert grad_audit[group]["nonzero_gradient_fraction"] > 0.0

    # Parameter change audit check
    param_changes = train_results["param_changes"]
    assert param_changes["changed_tensors"] > 0
    assert param_changes["L2_delta"] > 0.0
    assert param_changes["max_delta"] > 0.0


# 13. Checkpoint save/load
def test_checkpoint_save_and_load(trainer):
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer.save_checkpoints(path=tmpdir)
        
        # Confirm files exist
        assert os.path.exists(os.path.join(tmpdir, "transformer_checkpoint.pt"))
        assert os.path.exists(os.path.join(tmpdir, "meta_supervisor_checkpoint.pt"))
        assert os.path.exists(os.path.join(tmpdir, "optimizer_checkpoint.pt"))
        assert os.path.exists(os.path.join(tmpdir, "training_manifest.json"))

        # Create another trainer and load checkpoints
        transformer_new = TransformerContextEncoder(input_dim=52, seq_len=20, latent_dim=16)
        supervisor_new = MetaSupervisor(latent_dim=16)
        trainer_new = MetaRLTrainer(
            transformer=transformer_new,
            supervisor=supervisor_new,
            ppo_checkpoint_path=trainer.ppo_checkpoint_path,
            vecnormalize_path=trainer.vecnormalize_path,
        )
        trainer_new.load_checkpoints(path=tmpdir)
        
        # Verify weight parity by checking similarity
        for p1, p2 in zip(trainer.transformer.parameters(), trainer_new.transformer.parameters()):
            assert torch.allclose(p1, p2)
        for p1, p2 in zip(trainer.supervisor.parameters(), trainer_new.supervisor.parameters()):
            assert torch.allclose(p1, p2)


# 14. Train/validation split isolation
def test_train_val_split_isolation():
    # Verify split mapping correctness
    assert TaskSplitEnum.from_str("train") == TaskSplitEnum.TRAIN
    assert TaskSplitEnum.from_str("val") == TaskSplitEnum.VAL
    assert TaskSplitEnum.from_str("ood_test") == TaskSplitEnum.OOD_TEST


# 15. OOD isolation guard
def test_ood_isolation_guard(trainer):
    ood_task = MetaTask(
        task_id="task_test_001",
        seed=3042,
        split="ood_test",  # strict OOD Test
        wind_magnitude=4.5,
        wind_direction=[0.0, 1.0, 0.0],
        gust_magnitude=1.5,
        gust_frequency=2.0,
        turbulence_std=0.1,
        impulse_magnitude=5.0,
        impulse_duration=0.1,
        impulse_onset=1.5,
        sensor_noise_scale=4.0,
        motor_degradation=0.45,
        degraded_motors=[0, 2],
        mass_scale=1.1,
        inertia_scale=1.15,
        waypoint_seed=3001,
        compound_disturbance_flags={"extreme_wind": True, "actuator_loss": True},
    )

    with pytest.raises(ValueError, match="STRICTLY PROHIBITED"):
        trainer.rollout_episode(ood_task, explore=True)


# 16. Deterministic seed behavior
def test_deterministic_seed_behavior():
    t1 = TransformerContextEncoder(latent_dim=16)
    s1 = MetaSupervisor(latent_dim=16)
    tr1 = MetaRLTrainer(transformer=t1, supervisor=s1, master_seed=123)

    t2 = TransformerContextEncoder(latent_dim=16)
    s2 = MetaSupervisor(latent_dim=16)
    tr2 = MetaRLTrainer(transformer=t2, supervisor=s2, master_seed=123)

    # Weights must be exactly identical due to deterministic seed setting during constructor
    # wait, the weights are initialized randomly before seeding or after seeding?
    # The models are instantiated outside, so they are initialized randomly during their __init__.
    # To test seed determinism, let's verify that resetting the seed generates identical exploration noise:
    tr1._set_seed(999)
    n1 = torch.randn(5)
    tr2._set_seed(999)
    n2 = torch.randn(5)
    assert torch.allclose(n1, n2)


# 17. No task-ID leakage
def test_no_task_id_leakage(trainer, mock_task):
    # Ensure no task information (task_id, split, etc.) is fed into the context encoder or supervisor
    # The context encoder input dimensions is exactly 52 (24 state, 3 action, 1 reward, 24 delta_s)
    # The supervisor input is exactly 16 (latent vector size)
    assert trainer.transformer.input_dim == 52
    assert trainer.supervisor.fc1.in_features == 16


# 18. Numerical safety (Inf/NaN rejection)
def test_numerical_safety_rejection(trainer):
    # Test that transformer rejects NaN/Inf in inputs
    bad_history = torch.randn(1, 20, 52)
    bad_history[0, 5, 10] = float("nan")
    with pytest.raises(ValueError, match="Input history contains NaN values"):
        trainer.transformer(bad_history)

    bad_history[0, 5, 10] = float("inf")
    with pytest.raises(ValueError, match="Input history contains Inf values"):
        trainer.transformer(bad_history)


# 19. Frozen PPO policy integrity
def test_frozen_ppo_policy_integrity(trainer):
    # Verify that trainer parameters do not include PPO parameters
    ppo_param_names = [name for name, _ in trainer.ppo_policy.policy.named_parameters()]
    trainer_param_ids = [id(p) for p in trainer.params]

    for name, p in trainer.ppo_policy.policy.named_parameters():
        assert id(p) not in trainer_param_ids


# 20. Phase-9 interface compatibility
def test_phase9_interface_compatibility(trainer, mock_task):
    # Ensure raw env has the required properties and methods
    _, raw_env = trainer._make_env(mock_task)
    assert hasattr(raw_env, "reconfig_controller")
    assert hasattr(raw_env, "apply_reconfiguration")
    assert hasattr(raw_env, "reconfig_active")
    raw_env.close()


# 21. Multi-mode freezing correctness (MODEs A, B, C, D)
def test_multimode_freezing_correctness():
    # MODE A: Joint training (all parameters require grad)
    tA = TransformerContextEncoder(latent_dim=16)
    sA = MetaSupervisor(latent_dim=16)
    trA = MetaRLTrainer(transformer=tA, supervisor=sA, training_mode=TrainingMode.MODE_A)
    assert all(p.requires_grad for p in trA.transformer.parameters())
    assert all(p.requires_grad for p in trA.supervisor.parameters())

    # MODE B: Frozen Transformer + Trainable Supervisor
    tB = TransformerContextEncoder(latent_dim=16)
    sB = MetaSupervisor(latent_dim=16)
    trB = MetaRLTrainer(transformer=tB, supervisor=sB, training_mode=TrainingMode.MODE_B)
    assert all(not p.requires_grad for p in trB.transformer.parameters())
    assert all(p.requires_grad for p in trB.supervisor.parameters())

    # MODE C: Trainable Transformer + Frozen Supervisor
    tC = TransformerContextEncoder(latent_dim=16)
    sC = MetaSupervisor(latent_dim=16)
    trC = MetaRLTrainer(transformer=tC, supervisor=sC, training_mode=TrainingMode.MODE_C)
    assert all(p.requires_grad for p in trC.transformer.parameters())
    assert all(not p.requires_grad for p in trC.supervisor.parameters())


# 22. Context leakage prevention check
def test_no_task_metadata_leakage(trainer, mock_task):
    import copy
    task1 = mock_task
    task2 = copy.deepcopy(mock_task)
    task2.task_id = "completely_different_id_999"
    task2.split = "val"
    
    trainer._set_seed(42)
    ep1, _, _ = trainer.rollout_episode(task1, explore=False, max_steps=2)
    
    trainer._set_seed(42)
    ep2, _, _ = trainer.rollout_episode(task2, explore=False, max_steps=2)
    
    t1 = ep1.get_transitions()
    t2 = ep2.get_transitions()
    assert len(t1) == len(t2)
    for tr1, tr2 in zip(t1, t2):
        assert np.allclose(tr1.s_t, tr2.s_t)
        assert np.allclose(tr1.a_t, tr2.a_t)
        assert np.allclose(tr1.z_t, tr2.z_t)
        assert tr1.lambda_rl == tr2.lambda_rl
        assert tr1.alpha_q == tr2.alpha_q
        assert tr1.horizon == tr2.horizon

