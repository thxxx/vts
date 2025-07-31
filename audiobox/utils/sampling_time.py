import torch
import random
import numpy as np

def sample_lognorm(bs:int, device, dtype):
    if random.random()<0.4:
        t = torch.rand((bs, ), dtype=dtype, device=device)
    else:
        tnorm = np.random.normal(loc=0, scale=1.0, size=bs)
        t = 1 / (1 + np.exp(-tnorm))
        t = torch.tensor(t, dtype=dtype, device=device)
    return t
    # return torch.rand(bs, device=device, dtype=dtype)