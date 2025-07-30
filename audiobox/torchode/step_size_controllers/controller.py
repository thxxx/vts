import math
from typing import Generic, TypeVar

import torch
from einops import rearrange
from torch import Tensor, nn

from torchode.problems import InitialValueProblem
from torchode.single_step_methods.base import StepResult
from torchode.terms import ODETerm
from torchode.typing import (
    AcceptTensor,
    DataTensor,
    NormTensor,
    StatusTensor,
    TimeTensor,
)

ControllerState = TypeVar("ControllerState")


class StepSizeController(nn.Module, Generic[ControllerState]):
    """A step size controller determines the size of integration steps."""

    def init(
        self,
        term: ODETerm,
        problem: InitialValueProblem,
        method_order: int,
        *,
        stats: dict[str, Tensor],
    ) -> tuple[TimeTensor, ControllerState, DataTensor]:
        """Find the initial step size and initialize the controller state

        If the user suggests an initial step size, the controller should go with that
        one.

        If the controller evaluates the vector field at the initial step, for example
        to determine the initial step size, the evaluation at `t0` can be returned to
        save an evaluation in FSAL step methods.

        Arguments
        ---------
        term
            The integration term
        problem
            The problem to solve
        method_order
            Convergence order of the stepping method
        dt0
            An initial step size suggested by the user
        stats
            Container that the controller can initialize new statistics to track in
        """
        raise NotImplementedError()

    @property
    def adaptive(self) -> bool:
        raise NotImplementedError()

    def adapt_step_size(
        self,
        t0: TimeTensor,
        dt: TimeTensor,
        y0: DataTensor,
        step_result: StepResult,
        state: ControllerState,
        stats: dict[str, Tensor],
    ) -> tuple[AcceptTensor, TimeTensor, ControllerState, StatusTensor]:
        """Adapt the integration step size based on the step just taken

        Arguments
        ---------
        t0
            Start time of the step
        dt
            Current step size
        y0
            State before the previous step
        step_result
            Result of the step just taken
        state
            Current controller state
        stats
            Tracked statistics for the current solve which can be updated in-place

        Returns
        -------
        accept
            Should the step be accepted or rejected?
        dt
            Next step size, either for the next step or to retry the current step
            if it was rejected
        state
            Next controller state
        status
            Status to signify if integration should be stopped early (or None for
            all sucesses)
        """
        raise NotImplementedError()

    def merge_states(
        self, running: AcceptTensor, current: ControllerState, previous: ControllerState
    ) -> ControllerState:
        """Merge two controller states

        Any batch-specific state should be updated so that it updates only for the still
        running instances and stays constant for finished instances.

        Arguments
        ---------
        running
            Marks the instances in the batch that are still being solved
        current
            The controller state at the end of the current iteration
        previous
            The previous controller state
        """
        raise NotImplementedError()


class AdaptiveStepSizeController(
    StepSizeController[ControllerState], Generic[ControllerState]
):
    @property
    def adaptive(self) -> bool:
        return True

    def initial_state(
        self,
        method_order: int,
        problem: InitialValueProblem,
    ) -> ControllerState:
        raise NotImplementedError()

    def update_state(
        self,
        state: ControllerState,
        y0: DataTensor,
        dt: TimeTensor,
        error_ratio: NormTensor,
        accept: AcceptTensor,
    ) -> ControllerState:
        raise NotImplementedError()

    def dt_factor(self, state: ControllerState, error_ratio: NormTensor):
        raise NotImplementedError()


def rms_norm(y: DataTensor) -> NormTensor:
    """Root mean squared error norm.

    As suggested in [1], Equation (4.11).

    References
    ----------
    .. [1] E. Hairer, S. P. Norsett G. Wanner, "Solving Ordinary Differential
           Equations I: Nonstiff Problems", Sec. II.4, 2nd edition.
    """
    # `vector_norm` autmatically deals with complex vectors correctly
    y = rearrange(y, "b ... -> b (...)")
    return torch.linalg.vector_norm(y, ord=2, dim=1) / math.sqrt(y.shape[1])
