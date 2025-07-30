import torch
from torch import Tensor, nn


class Snake1d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x: Tensor):
        x = x + x.clone().mul_(self.alpha).sin_().pow_(2).div_(self.alpha + 1e-7)
        return x
