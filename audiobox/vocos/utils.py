import torch
from einops import rearrange
from torch import Tensor


def mask_from_lengths(lengths: Tensor, max_len: int) -> Tensor:
    seq = torch.arange(max_len, device=lengths.device)
    mask = seq < rearrange(lengths, "b -> b ()")
    return mask
