"""MCR-UAV Standalone Transformer Context Encoder Architecture.

This module implements the temporal Transformer Context Encoder ($f_\\phi$)
for the Meta-Contextual Reconfiguration (MCR-UAV) framework.

Mathematical Formulation:
- Input history sequence: H_t = [x_{t-L+1}, ..., x_t] \\in \\mathbb{R}^{B \\times L \\times d_h}
- Transition tuple: x_t = [s_t, a_t, r_t, \\Delta s_t] \\in \\mathbb{R}^{52}
  where s_t \\in \\mathbb{R}^{24}, a_t \\in \\mathbb{R}^3, r_t \\in \\mathbb{R}^1, \\Delta s_t \\in \\mathbb{R}^{24}
- Linear projection: 52 \\to 64
- Sinusoidal Positional Encoding (L_max = 20)
- Multi-Head Causal Self-Attention: d_model = 64, n_heads = 4, n_layers = 2
- Causal Mask: Upper-triangular blocking (j > i blocked)
- Latent Projection: 64 \\to 16 \\implies z_t \\in \\mathbb{R}^{B \\times 16}
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """Deterministic Sinusoidal Positional Encoding for temporal transition histories."""

    def __init__(self, d_model: int = 64, max_len: int = 20) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input embeddings.
        
        Parameters
        ----------
        x : torch.Tensor
            Projected input tensor of shape [B, L, d_model].
            
        Returns
        -------
        torch.Tensor
            Position-encoded tensor of shape [B, L, d_model].
        """
        seq_len = x.size(1)
        if seq_len > self.max_len:
            raise ValueError(
                f"Input sequence length {seq_len} exceeds maximum configured length {self.max_len}"
            )
        return x + self.pe[:, :seq_len]


class TransformerContextEncoder(nn.Module):
    """Standalone Temporal Transformer Context Encoder for MCR-UAV."""

    def __init__(
        self,
        input_dim: int = 52,
        seq_len: int = 20,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        latent_dim: int = 16,
        dropout: float = 0.0,
        activation: str = "gelu",
        use_layer_norm: bool = True,
    ) -> None:
        """Initialize the Transformer Context Encoder.
        
        Parameters
        ----------
        input_dim : int
            Dimension of transition tuple x_t (default: 52).
        seq_len : int
            Maximum history window length L (default: 20).
        d_model : int
            Transformer embedding dimension (default: 64).
        n_heads : int
            Number of attention heads (default: 4).
        n_layers : int
            Number of Transformer encoder layers (default: 2).
        latent_dim : int
            Dimension of output latent context z_t (default: 16).
        dropout : float
            Dropout probability (default: 0.0).
        activation : str
            Activation function in feedforward network ('gelu' or 'relu').
        use_layer_norm : bool
            Whether to apply LayerNorm before final latent projection (default: True).
        """
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.latent_dim = latent_dim
        self.dropout = dropout

        # 1. Input Linear Projection
        self.input_proj = nn.Linear(input_dim, d_model)

        # 2. Sinusoidal Positional Encoding
        self.pos_encoder = SinusoidalPositionalEncoding(d_model=d_model, max_len=seq_len)

        # 3. Transformer Encoder Layers with Pre-LN & Causal Masking
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # 4. Pre-Projection LayerNorm & Latent Output Projection
        self.layer_norm = nn.LayerNorm(d_model) if use_layer_norm else nn.Identity()
        self.latent_proj = nn.Linear(d_model, latent_dim)

        # 5. Persistent Causal Mask Buffer (Upper-triangular blocked: True = masked)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask", causal_mask)

    def _validate_input(self, history: torch.Tensor) -> None:
        """Validate input tensor sanity (finite values, dimensional consistency)."""
        if not torch.is_floating_point(history):
            raise TypeError(f"History tensor must be floating point, got {history.dtype}")

        if torch.isnan(history).any():
            raise ValueError("Input history contains NaN values; rejecting numerical corruption")

        if torch.isinf(history).any():
            raise ValueError("Input history contains Inf values; rejecting numerical explosion")

        if history.dim() == 2:
            if history.size(-1) != self.input_dim:
                raise ValueError(
                    f"Expected transition dimension {self.input_dim}, got {history.size(-1)}"
                )
        elif history.dim() == 3:
            if history.size(-1) != self.input_dim:
                raise ValueError(
                    f"Expected transition dimension {self.input_dim}, got {history.size(-1)}"
                )
        else:
            raise ValueError(
                f"Expected 2D [L, d_h] or 3D [B, L, d_h] history tensor, got shape {history.shape}"
            )

    def forward(
        self,
        history: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        valid_lens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass computing the latent context vector z_t.
        
        Parameters
        ----------
        history : torch.Tensor
            Transition history tensor of shape [B, L, 52] or [L, 52].
        padding_mask : Optional[torch.Tensor]
            Boolean tensor of shape [B, L] where True indicates padded tokens to ignore.
        valid_lens : Optional[torch.Tensor]
            1D integer tensor [B] indicating the number of valid transitions per batch item.
            
        Returns
        -------
        torch.Tensor
            Latent context representation z_t of shape [B, 16].
        """
        # Validate numerical inputs
        self._validate_input(history)

        # Handle 2D unbatched input [L, 52] -> [1, L, 52]
        is_unbatched = history.dim() == 2
        if is_unbatched:
            history = history.unsqueeze(0)
            if padding_mask is not None and padding_mask.dim() == 1:
                padding_mask = padding_mask.unsqueeze(0)
            if valid_lens is not None and valid_lens.dim() == 0:
                valid_lens = valid_lens.unsqueeze(0)

        batch_size, seq_len, _ = history.shape

        # 1. Linear Input Projection [B, L, 52] -> [B, L, 64]
        x = self.input_proj(history)

        # 2. Add Positional Encoding
        x = self.pos_encoder(x)

        # 3. Dynamic Causal Attention Masking
        causal_mask = self.causal_mask[:seq_len, :seq_len]

        # 4. Transformer Forward Pass with Causal & Padding Mask
        # PyTorch src_key_padding_mask: True means padded/ignored
        h_seq = self.transformer(
            src=x,
            mask=causal_mask,
            src_key_padding_mask=padding_mask,
        )  # Shape: [B, L, 64]

        # 5. Extract representation at the final valid timestep for each sequence
        if valid_lens is not None:
            # Clamp valid lengths to valid sequence range [1, seq_len]
            clamped_lens = torch.clamp(valid_lens.long(), min=1, max=seq_len)
            last_indices = clamped_lens - 1
            # Gather representation: [B, 1, 64]
            last_rep = h_seq[torch.arange(batch_size, device=history.device), last_indices]
        elif padding_mask is not None:
            # Derive valid lengths from padding mask (~padding_mask count per row)
            valid_counts = (~padding_mask).long().sum(dim=1)
            valid_counts = torch.clamp(valid_counts, min=1, max=seq_len)
            last_indices = valid_counts - 1
            last_rep = h_seq[torch.arange(batch_size, device=history.device), last_indices]
        else:
            # Default: take the last token representation at index L-1
            last_rep = h_seq[:, -1, :]  # Shape: [B, 64]

        # 6. Apply LayerNorm & Project to Latent Dimension: [B, 64] -> [B, 16]
        normed = self.layer_norm(last_rep)
        z = self.latent_proj(normed)  # Shape: [B, 16]

        return z

    def get_sequence_representations(
        self,
        history: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Extract full causal sequence representations for temporal causality verification.
        
        Parameters
        ----------
        history : torch.Tensor
            Tensor of shape [B, L, 52].
        padding_mask : Optional[torch.Tensor]
            Tensor of shape [B, L].
            
        Returns
        -------
        torch.Tensor
            Full sequence representations of shape [B, L, 64].
        """
        self._validate_input(history)
        if history.dim() == 2:
            history = history.unsqueeze(0)
            if padding_mask is not None and padding_mask.dim() == 1:
                padding_mask = padding_mask.unsqueeze(0)

        _, seq_len, _ = history.shape
        x = self.input_proj(history)
        x = self.pos_encoder(x)
        causal_mask = self.causal_mask[:seq_len, :seq_len]
        h_seq = self.transformer(src=x, mask=causal_mask, src_key_padding_mask=padding_mask)
        return h_seq

    def count_parameters(self) -> int:
        """Return total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
