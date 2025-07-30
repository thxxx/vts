import torch

def sample_lognorm(bs:int, device, dtype):
    return torch.rand(bs, device=device, dtype=dtype)