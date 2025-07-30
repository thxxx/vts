from collections.abc import Callable
from typing import Final

from torch import Tensor, nn

from torchode.typing import DataTensor, TimeTensor


class ODETerm(nn.Module):
    with_stats: Final[bool]

    def __init__(
        self,
        f: Callable[[TimeTensor, DataTensor], DataTensor],
        *,
        with_stats: bool = True,
    ):
        """Initialize an ODE term of the form `dy/dt = f(t, y)`.

        Arguments
        ---------
        f
            Right-hand side of the ODE
        with_stats
            If true, track statistics such as the number of function evaluations. If your
            dynamics are very fast to evaluate, disabling this can improve the performance
            of the solver by 1-2%.
        """

        super().__init__()

        self.f = f
        self.with_stats = with_stats

    def vf(self, t: TimeTensor, y: DataTensor, stats: dict[str, Tensor]) -> DataTensor:
        """Evaluate the vector field."""
        if self.with_stats:
            n_f_evals = stats["n_f_evals"]
            n_f_evals.add_(1)

        return self.f(t, y)
