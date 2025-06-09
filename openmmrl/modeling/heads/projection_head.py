import flax.linen as nn
import jax.numpy as jnp

import torch

class ProjectionHead(nn.Module):
    """
    A Flax-based projection head (MLP).
    """
    input_dim: int
    hidden_dim: int
    output_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.output_dim)(x)
        return x

class ProjectionHeadPyTorch(torch.nn.Module):
    """
    A PyTorch-based projection head (MLP).
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.mlp(x) 