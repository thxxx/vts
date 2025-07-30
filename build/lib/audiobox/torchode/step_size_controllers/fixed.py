from typing import Any, NamedTuple

import torch

from torchode import status_codes
from torchode.problems import InitialValueProblem
from torchode.single_step_methods.base import StepResult
from torchode.step_size_controllers.controller import (
    AdaptiveStepSizeController,
)
from torchode.terms import ODETerm
from torchode.typing import (
    AcceptTensor,
    DataTensor,
    StatusTensor,
    TimeTensor,
)


class FixedStepState(NamedTuple):
    accept_all: AcceptTensor
    dt0: TimeTensor


class FixedStepController(AdaptiveStepSizeController[FixedStepState]):
    """A fixed-step step size controller.

    Does not actually control anything. Just accepts any result and keeps the step size
    fixed.
    """

    @property
    def adaptive(self) -> bool:
        return False

    def init(
        self,
        term: ODETerm,
        problem: InitialValueProblem,
        method_order: int,
        *,
        stats: dict[str, Any],
    ):
        return (
            problem.t_eval[:, 1],
            FixedStepState(
                accept_all=torch.ones(
                    problem.batch_size, device=problem.device, dtype=torch.bool
                ),
                dt0=problem.t_eval[:, 1],
            ),
            term.vf(problem.t_start, problem.y0, stats),
        )

    def adapt_step_size(
        self,
        t0: TimeTensor,
        dt: TimeTensor,
        y0: DataTensor,
        step_result: StepResult,
        state: FixedStepState,
        stats: dict[str, Any],
    ) -> tuple[AcceptTensor, TimeTensor, FixedStepState, StatusTensor]:
        return (
            state.accept_all,
            state.dt0,
            state,
            torch.full_like(
                step_result.error_estimate, status_codes.SUCCESS, dtype=torch.long
            ),
        )

    def merge_states(
        self, running: AcceptTensor, current: FixedStepState, previous: FixedStepState
    ) -> FixedStepState:
        return current
