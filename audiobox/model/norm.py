import torch
from torch import Tensor, nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (self.dim,), self.gamma)
