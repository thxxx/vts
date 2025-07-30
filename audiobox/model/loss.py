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

    # shape: (B, L)
    loss = reduce(loss, "b l c -> b l", "mean")
    loss = loss * mask

    # 안정화
    mask_sum = reduce(mask, "b l -> b", "sum")
    mask_sum = mask_sum.clamp(min=1.0)  # NaN 방지

    loss = reduce(loss, "b l -> b", "sum") / mask_sum
    return loss.mean()