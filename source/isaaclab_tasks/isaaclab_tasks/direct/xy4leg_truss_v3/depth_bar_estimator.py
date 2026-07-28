from __future__ import annotations

import torch
from torch import nn


class DepthBarEstimator(nn.Module):
    """Small CNN that predicts metric body-frame crossbar displacement from a local heightmap."""

    def __init__(
        self,
        input_channels: int = 2,
        output_scale: tuple[float, float, float] = (1.2, 0.3, 0.5),
    ):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=5, stride=2, padding=2),
            nn.ELU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(64, 64),
            nn.ELU(),
            nn.Linear(64, 3),
        )
        self.register_buffer("output_scale", torch.tensor(output_scale, dtype=torch.float32))

    def forward(self, heightmap: torch.Tensor) -> torch.Tensor:
        # heightmap: [N, 2, H, W], channel 0 is normalized height, channel 1 is valid mask.
        heightmap = torch.nan_to_num(heightmap, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        return torch.tanh(self.head(self.features(heightmap))) * self.output_scale
