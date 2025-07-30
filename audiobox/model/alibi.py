import torch
from einops import einsum, rearrange
from torch import Tensor


def get_alibi_bias(nhead: int, max_length: int, device: torch.device) -> Tensor:
    arange = torch.arange(max_length, device=device)
    rel = rearrange(arange, "l -> l ()") - rearrange(arange, "l -> () l")
    slope = -torch.arange(1, nhead + 1, device=device) * 8 / nhead
    bias = einsum(rel, slope.exp2(), "q k, h -> h q k")
    return -bias.abs()
