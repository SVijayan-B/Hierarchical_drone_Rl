"""Comprehensive Unit Tests for MCR-UAV Transformer Context Encoder Architecture."""

import os
import sys
import pytest
import torch
import torch.nn as nn
import numpy as np

# Ensure root workspace is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.transformer_context_encoder import (
    SinusoidalPositionalEncoding,
    TransformerContextEncoder,
)


def test_model_architecture_and_parameter_count():
    """Verify architectural hyperparameters and report parameter count."""
    model = TransformerContextEncoder(
        input_dim=52,
        seq_len=20,
        d_model=64,
        n_heads=4,
        n_layers=2,
        latent_dim=16,
        dropout=0.0,
    )
    param_count = model.count_parameters()
    assert model.input_dim == 52
    assert model.seq_len == 20
    assert model.d_model == 64
    assert model.n_heads == 4
    assert model.n_layers == 2
    assert model.latent_dim == 16
    assert param_count > 0
    print(f"\n[Architecture Test] Total Trainable Parameters: {param_count:,}")


@pytest.mark.parametrize("batch_size", [1, 4, 8, 16, 32])
def test_latent_output_shapes_and_batch_sizes(batch_size):
    """Test output shape [B, 16] across various batch sizes."""
    model = TransformerContextEncoder()
    model.eval()
    
    x = torch.randn(batch_size, 20, 52)
    with torch.no_grad():
        z = model(x)
        
    assert z.shape == (batch_size, 16)
    assert torch.isfinite(z).all()


def test_single_sample_unbatched_input():
    """Test passing unbatched [20, 52] input returns [1, 16]."""
    model = TransformerContextEncoder()
    model.eval()
    
    x = torch.randn(20, 52)
    with torch.no_grad():
        z = model(x)
        
    assert z.shape == (1, 16)
    assert torch.isfinite(z).all()


def test_gradient_flow_and_trainability():
    """Test that all parameters receive finite non-zero gradients on backprop."""
    model = TransformerContextEncoder()
    model.train()
    
    x = torch.randn(4, 20, 52, requires_grad=True)
    z = model(x)
    loss = z.sum()
    loss.backward()
    
    # Check input gradients
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    
    # Check all model parameter gradients
    for name, p in model.named_parameters():
        assert p.grad is not None, f"Parameter {name} has no gradient"
        assert torch.isfinite(p.grad).all(), f"Parameter {name} has non-finite gradient"
        assert (p.grad.abs().sum() > 0.0), f"Parameter {name} has zero gradient"


def test_strict_causality_no_future_information_leakage():
    """CRITICAL TEST: Verify that future transitions cannot affect earlier representations."""
    model = TransformerContextEncoder()
    model.eval()
    
    torch.manual_seed(42)
    # Base history of 20 transitions
    h1 = torch.randn(1, 20, 52)
    
    # Perturbed history: identical for t = 0..9, completely different for t = 10..19
    h2 = h1.clone()
    h2[:, 10:, :] = torch.randn(1, 10, 52) * 5.0 + 10.0
    
    with torch.no_grad():
        # Get full sequence representations
        seq_rep1 = model.get_sequence_representations(h1)  # [1, 20, 64]
        seq_rep2 = model.get_sequence_representations(h2)  # [1, 20, 64]
        
        # Representations at t = 0..9 must be IDENTICAL (no future leakage from t >= 10)
        past_diff = (seq_rep1[:, :10, :] - seq_rep2[:, :10, :]).abs().max().item()
        assert past_diff < 1e-5, f"Causality violation! Future leaked into past with max diff {past_diff}"
        
        # Representations at t >= 10 MUST differ because input at t >= 10 changed
        future_diff = (seq_rep1[:, 10:, :] - seq_rep2[:, 10:, :]).abs().max().item()
        assert future_diff > 1e-2, "Future representation unexpectedly identical"
        
        # Latent context computed at timestep t = 10 must be IDENTICAL on both histories
        z1 = model(h1, valid_lens=torch.tensor([10]))
        z2 = model(h2, valid_lens=torch.tensor([10]))
        latent_diff = (z1 - z2).abs().max().item()
        assert latent_diff < 1e-5, f"Latent context causality violated! Diff: {latent_diff}"


@pytest.mark.parametrize("valid_len", [1, 5, 10, 20])
def test_variable_history_lengths_and_padding_masks(valid_len):
    """Test variable history lengths using explicit valid_lens and padding_mask."""
    model = TransformerContextEncoder()
    model.eval()
    
    batch_size = 4
    h = torch.randn(batch_size, 20, 52)
    
    # Create boolean padding mask: True indicates padded (ignored) tokens
    padding_mask = torch.zeros(batch_size, 20, dtype=torch.bool)
    padding_mask[:, valid_len:] = True
    
    # Fill padded positions with arbitrary large noise
    h_padded = h.clone()
    h_padded[:, valid_len:, :] = torch.randn(batch_size, 20 - valid_len, 52) * 100.0
    
    valid_lens = torch.full((batch_size,), valid_len, dtype=torch.long)
    
    with torch.no_grad():
        z_valid_lens = model(h_padded, valid_lens=valid_lens)
        z_padding_mask = model(h_padded, padding_mask=padding_mask)
        
    assert z_valid_lens.shape == (batch_size, 16)
    assert z_padding_mask.shape == (batch_size, 16)
    assert torch.isfinite(z_valid_lens).all()
    assert torch.isfinite(z_padding_mask).all()
    
    # Both methods of specifying sequence length should produce identical latent vectors
    diff = (z_valid_lens - z_padding_mask).abs().max().item()
    assert diff < 1e-5, f"Mismatch between valid_lens and padding_mask: {diff}"


def test_numerical_assertions_nan_and_inf_detection():
    """Test that NaN and Inf inputs are strictly detected and rejected."""
    model = TransformerContextEncoder()
    model.eval()
    
    # NaN input
    x_nan = torch.randn(2, 20, 52)
    x_nan[0, 5, 10] = float("nan")
    with pytest.raises(ValueError, match="NaN"):
        model(x_nan)
        
    # Inf input
    x_inf = torch.randn(2, 20, 52)
    x_inf[1, 2, 3] = float("inf")
    with pytest.raises(ValueError, match="Inf"):
        model(x_inf)
        
    # Dimension mismatch
    x_wrong_dim = torch.randn(2, 20, 50)
    with pytest.raises(ValueError, match="Expected transition dimension 52"):
        model(x_wrong_dim)


def test_deterministic_eval_mode():
    """Test deterministic evaluation mode produces exact bitwise identity."""
    model = TransformerContextEncoder()
    model.eval()
    
    x = torch.randn(4, 20, 52)
    with torch.no_grad():
        z1 = model(x)
        z2 = model(x)
        
    assert torch.equal(z1, z2)
