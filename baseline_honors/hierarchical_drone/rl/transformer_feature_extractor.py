import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class TransformerFeatureExtractor(BaseFeaturesExtractor):
    """Custom SB3 Feature Extractor using a Transformer Encoder over stacked observation histories."""

    def __init__(
        self,
        observation_space,
        features_dim: int = 128,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2
    ):
        # The observation space is a flat array representing [20 * obs_dim]
        total_dim = observation_space.shape[0]
        self.history_len = 20
        self.obs_dim = total_dim // self.history_len
        
        super().__init__(observation_space, features_dim)
        
        self.embed_dim = embed_dim
        
        # Projection of input dimension to embedding dimension
        self.input_projection = nn.Linear(self.obs_dim, embed_dim)
        
        # Learnable Positional Encoding [1, 20, embed_dim]
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.history_len, embed_dim))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        
        # Transformer Encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=0.0,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        # Output mapping
        self.fc = nn.Linear(embed_dim, features_dim)
        self.relu = nn.ReLU()

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # observations shape: [batch_size, 20 * obs_dim]
        batch_size = observations.shape[0]
        
        # Reshape to [batch_size, 20, obs_dim]
        x = observations.view(batch_size, self.history_len, self.obs_dim)
        
        # Project inputs to embed_dim
        x = self.input_projection(x)  # [batch_size, 20, embed_dim]
        
        # Add Positional Encoding
        x = x + self.pos_embedding  # Broadcasts across batch size
        
        # Pass through Transformer Encoder
        x = self.transformer_encoder(x)  # [batch_size, 20, embed_dim]
        
        # Extract features for the last step in the history sequence
        x_last = x[:, -1, :]  # [batch_size, embed_dim]
        
        # Apply Layer Norm and projection
        x_norm = self.layer_norm(x_last)
        out = self.relu(self.fc(x_norm))  # [batch_size, features_dim]
        
        return out
