from einops import reduce
from torch.nn import functional as F

from utils.typing import EncMaskTensor, EncTensor, LossTensor


def masked_loss(
    x: EncTensor, target: EncTensor, mask: EncMaskTensor, loss_method: str
) -> LossTensor:
    match loss_method:
        case "mse":
            loss = F.mse_loss(x, target, reduction="none")
        case "mae":
            loss = F.l1_loss(x, target, reduction="none")
        case _:
            raise ValueError(f"loss method {loss_method} not supported")

    loss = reduce(loss, "b l c -> b l", "mean")
    loss = loss * mask
    loss = reduce(loss, "b l -> b", "sum") / reduce(mask, "b l -> b", "sum")
    return loss.mean()
