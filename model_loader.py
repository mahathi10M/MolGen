"""
model_loader.py — loads the trained DiffusionModel from model.pth.

The architecture here MUST match the one used during training (molgen.ipynb).
Do not use the old SimpleDiffusionModel / fc1-fc2 classifier — that is a
completely different architecture and will raise a state-dict key mismatch error.
"""

import math
import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    """Positional-style embedding for diffusion timesteps."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half  = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
        )
        args = t[:, None].float() * freqs[None]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class DiffusionModel(nn.Module):
    def __init__(self, input_dim: int = 2048, hidden_dim: int = 1024, time_dim: int = 128):
        super().__init__()
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 2), nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim),
        )
        self.net = nn.Sequential(
            nn.Linear(input_dim + time_dim, hidden_dim), nn.SiLU(), nn.LayerNorm(hidden_dim), nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),            nn.SiLU(), nn.LayerNorm(hidden_dim), nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),       nn.SiLU(), nn.LayerNorm(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, input_dim),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, self.time_embed(t)], dim=-1))


def load_model(weights_path: str = "model.pth") -> DiffusionModel:
    """Load the trained diffusion model from a .pth weights file."""
    device = torch.device("cpu")
    model  = DiffusionModel().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    print(f"✅ DiffusionModel loaded from {weights_path}")
    return model