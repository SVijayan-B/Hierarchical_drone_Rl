"""Comprehensive Unit Tests for MCR-UAV Meta-Supervisor Architecture."""

import io
import os
import sys
import pytest
import torch
import torch.nn as nn
import numpy as np

# Ensure root workspace is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.meta_supervisor import (
    MetaSupervisor,
    ReconfigurationBounds,
    ReconfigurationVector,
)


def test_architecture_configuration():
    """Verify architectural parameters and head initializations."""
    supervisor = MetaSupervisor(latent_dim=16, hidden_dim=64)
    assert supervisor.latent_dim == 16
    assert supervisor.hidden_dim == 64
    assert supervisor.num_horizons == 3
    assert supervisor.bounds.horizon_options == (10, 20, 30)
    assert supervisor.bounds.lambda_rl_min == 0.0
    assert supervisor.bounds.lambda_rl_max == 1.0


def test_parameter_count_and_breakdown():
    """Verify and report total trainable parameter count and layer breakdown."""
    supervisor = MetaSupervisor(latent_dim=16, hidden_dim=64)
    breakdown = supervisor.parameter_breakdown()
    total_params = supervisor.count_parameters()
    
    assert total_params > 0
    assert breakdown["total"] == total_params
    assert breakdown["input_network"] == (16 * 64 + 64) + (64 * 2) + (64 * 64 + 64)  # 1088 + 128 + 4160 = 5376
    assert breakdown["lambda_head"] == 64 * 1 + 1  # 65
    assert breakdown["alpha_q_head"] == 64 * 1 + 1  # 65
    assert breakdown["alpha_r_head"] == 64 * 1 + 1  # 65
    assert breakdown["alpha_p_head"] == 64 * 1 + 1  # 65
    assert breakdown["alpha_i_head"] == 64 * 1 + 1  # 65
    assert breakdown["alpha_d_head"] == 64 * 1 + 1  # 65
    assert breakdown["horizon_head"] == 64 * 3 + 3  # 195
    assert total_params == 5376 + 65 * 6 + 195  # 5961
    print(f"\n[MetaSupervisor Architecture] Total Trainable Parameters: {total_params:,}")


@pytest.mark.parametrize("batch_size", [1, 4, 8, 16, 32])
def test_batch_sizes_and_output_shapes(batch_size):
    """Test output shapes across various batch sizes B in {1, 4, 8, 16, 32}."""
    supervisor = MetaSupervisor()
    supervisor.eval()
    
    z = torch.randn(batch_size, 16)
    with torch.no_grad():
        out = supervisor(z)
        
    assert out.lambda_rl.shape == (batch_size, 1)
    assert out.alpha_q.shape == (batch_size, 1)
    assert out.alpha_r.shape == (batch_size, 1)
    assert out.alpha_p.shape == (batch_size, 1)
    assert out.alpha_i.shape == (batch_size, 1)
    assert out.alpha_d.shape == (batch_size, 1)
    assert out.horizon_logits.shape == (batch_size, 3)
    assert out.horizon_probabilities.shape == (batch_size, 3)
    assert out.horizon.shape == (batch_size, 1)
    assert out.continuous_vector.shape == (batch_size, 6)


def test_single_sample_unbatched_1d_input():
    """Test passing unbatched [16] latent context tensor."""
    supervisor = MetaSupervisor()
    supervisor.eval()
    
    z = torch.randn(16)
    with torch.no_grad():
        out = supervisor(z)
        
    assert out.lambda_rl.shape == (1, 1)
    assert out.continuous_vector.shape == (1, 6)
    assert out.horizon.shape == (1, 1)
    assert torch.isfinite(out.continuous_vector).all()


def test_deterministic_inference():
    """Test deterministic evaluation mode produces exact bitwise identity."""
    supervisor = MetaSupervisor()
    supervisor.eval()
    
    z = torch.randn(4, 16)
    with torch.no_grad():
        out1 = supervisor(z)
        out2 = supervisor(z)
        
    assert torch.equal(out1.continuous_vector, out2.continuous_vector)
    assert torch.equal(out1.horizon_logits, out2.horizon_logits)
    assert torch.equal(out1.horizon, out2.horizon)


def test_gradient_flow_continuous_and_discrete_heads():
    """Test that all weights receive non-zero, finite gradients on backprop."""
    supervisor = MetaSupervisor()
    supervisor.train()
    
    z = torch.randn(4, 16, requires_grad=True)
    out = supervisor(z)
    
    # Loss combining continuous parameters and horizon logits
    loss_cont = out.continuous_vector.sum()
    loss_disc = out.horizon_logits.sum()
    loss = loss_cont + loss_disc
    loss.backward()
    
    # Check input gradient
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()
    
    # Check all model parameter gradients
    for name, p in supervisor.named_parameters():
        assert p.grad is not None, f"Parameter {name} has no gradient"
        assert torch.isfinite(p.grad).all(), f"Parameter {name} has non-finite gradient"
        assert (p.grad.abs().sum() > 0.0), f"Parameter {name} has zero gradient"


def test_nan_input_rejection():
    """Test that NaN latent input is strictly rejected."""
    supervisor = MetaSupervisor()
    supervisor.eval()
    
    z_nan = torch.randn(2, 16)
    z_nan[0, 5] = float("nan")
    with pytest.raises(ValueError, match="NaN"):
        supervisor(z_nan)


def test_inf_input_rejection():
    """Test that Inf latent input is strictly rejected."""
    supervisor = MetaSupervisor()
    supervisor.eval()
    
    z_inf = torch.randn(2, 16)
    z_inf[1, 3] = float("inf")
    with pytest.raises(ValueError, match="Inf"):
        supervisor(z_inf)


def test_feature_dim_mismatch_rejection():
    """Test that mismatched feature dimension raises ValueError."""
    supervisor = MetaSupervisor(latent_dim=16)
    supervisor.eval()
    
    z_bad = torch.randn(2, 20)
    with pytest.raises(ValueError, match="Expected latent dimension 16"):
        supervisor(z_bad)


def test_continuous_parameter_bound_compliance():
    """Verify all continuous parameters strictly obey configured physical bounds."""
    bounds = ReconfigurationBounds(
        lambda_rl_min=0.0, lambda_rl_max=1.0,
        alpha_q_min=0.2, alpha_q_max=5.0,
        alpha_r_min=0.2, alpha_r_max=5.0,
        alpha_p_min=0.5, alpha_p_max=2.0,
        alpha_i_min=0.2, alpha_i_max=2.5,
        alpha_d_min=0.5, alpha_d_max=2.0,
    )
    supervisor = MetaSupervisor(bounds=bounds)
    supervisor.eval()
    
    # Evaluate across 100 random latent vectors
    torch.manual_seed(42)
    z = torch.randn(100, 16) * 10.0
    with torch.no_grad():
        out = supervisor(z)
        
    assert (out.lambda_rl >= 0.0).all() and (out.lambda_rl <= 1.0).all()
    assert (out.alpha_q >= 0.2).all() and (out.alpha_q <= 5.0).all()
    assert (out.alpha_r >= 0.2).all() and (out.alpha_r <= 5.0).all()
    assert (out.alpha_p >= 0.5).all() and (out.alpha_p <= 2.0).all()
    assert (out.alpha_i >= 0.2).all() and (out.alpha_i <= 2.5).all()
    assert (out.alpha_d >= 0.5).all() and (out.alpha_d <= 2.0).all()


def test_extreme_latent_boundary_robustness():
    """Test extreme latent inputs: zeros, +1000, -1000."""
    supervisor = MetaSupervisor()
    supervisor.eval()
    
    z_zero = torch.zeros(5, 16)
    z_pos = torch.full((5, 16), 1000.0)
    z_neg = torch.full((5, 16), -1000.0)
    
    for z in [z_zero, z_pos, z_neg]:
        with torch.no_grad():
            out = supervisor(z)
            
        assert torch.isfinite(out.continuous_vector).all()
        assert torch.isfinite(out.horizon_logits).all()
        assert (out.lambda_rl >= 0.0).all() and (out.lambda_rl <= 1.0).all()
        assert (out.alpha_q >= 0.2).all() and (out.alpha_q <= 5.0).all()
        assert (out.alpha_r >= 0.2).all() and (out.alpha_r <= 5.0).all()
        assert (out.alpha_p >= 0.5).all() and (out.alpha_p <= 2.0).all()
        assert (out.alpha_i >= 0.2).all() and (out.alpha_i <= 2.5).all()
        assert (out.alpha_d >= 0.5).all() and (out.alpha_d <= 2.0).all()


def test_horizon_categorical_validity():
    """Verify that discrete horizon selection H is strictly in {10, 20, 30}."""
    supervisor = MetaSupervisor()
    supervisor.eval()
    
    torch.manual_seed(123)
    z = torch.randn(500, 16) * 5.0
    with torch.no_grad():
        out = supervisor(z, return_hard=True)
        
    horizons = out.horizon.squeeze().tolist()
    valid_set = {10, 20, 30}
    for h in horizons:
        assert h in valid_set, f"Invalid horizon value generated: {h}"


def test_serialization_and_state_dict_consistency():
    """Verify save/load state_dict produces exact identical outputs."""
    supervisor1 = MetaSupervisor()
    supervisor1.eval()
    
    # Save state dict
    buffer = io.BytesIO()
    torch.save(supervisor1.state_dict(), buffer)
    buffer.seek(0)
    
    # Load into fresh instance
    supervisor2 = MetaSupervisor()
    supervisor2.load_state_dict(torch.load(buffer))
    supervisor2.eval()
    
    z = torch.randn(8, 16)
    with torch.no_grad():
        out1 = supervisor1(z)
        out2 = supervisor2(z)
        
    assert torch.equal(out1.continuous_vector, out2.continuous_vector)
    assert torch.equal(out1.horizon_logits, out2.horizon_logits)
    assert torch.equal(out1.horizon, out2.horizon)


def test_to_numpy_export_helper():
    """Test ReconfigurationVector to_numpy helper."""
    supervisor = MetaSupervisor()
    supervisor.eval()
    
    z = torch.randn(2, 16)
    with torch.no_grad():
        out = supervisor(z)
        np_dict = out.to_numpy()
        
    assert isinstance(np_dict["lambda_rl"], np.ndarray)
    assert isinstance(np_dict["horizon"], np.ndarray)
    assert np_dict["continuous_vector"].shape == (2, 6)
