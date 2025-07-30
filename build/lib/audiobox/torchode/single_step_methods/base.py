from typing import Generic, NamedTuple, TypeVar

from torch import Tensor, nn

from torchode.interpolation import LocalInterpolation
from torchode.problems import InitialValueProblem
from torchode.terms import ODETerm
from torchode.typing import AcceptTensor, DataTensor, TimeTensor


class StepResult(NamedTuple):
    y: DataTensor
    error_estimate: DataTensor


MethodState = TypeVar("MethodState")
InterpolationData = TypeVar("InterpolationData")


class SingleStepMethod(nn.Module, Generic[MethodState, InterpolationData]):
    def init(
        self,
        term: ODETerm,
        problem: InitialValueProblem,
        f0: DataTensor,
        *,
        stats: dict[str, Tensor],
    ) -> MethodState:
        raise NotImplementedError()

    def step(
        self,
        term: ODETerm,
        y0: DataTensor,
        t0: TimeTensor,
        dt: TimeTensor,
        state: MethodState,
        *,
        stats: dict[str, Tensor],
    ) -> tuple[StepResult, InterpolationData, MethodState]:
        """Advance the solution from `y0` to `y0+dt`.

        Arguments
        ---------
        running
            Marks the instances in the batch that are actually still running. This is
            important for solvers with variable computation time such as implicit methods
            that use this information to short-circuit the evaluation of finished
            instances.
        y0
            Features at `t0`
        t0
            Initial point in time
        dt
            Step size of the step to make
        state
            Current state of the stepping method
        stats
            Tracked statistics for the current solve

        Returns
        -------
        result
            Features `y1` at `t1 = t0 + dt` with an error estimate
        interpolation_data
            Additional information for interpolation between `t0` and `t1`
        state
            Updated state of the stepping method
        status
            Status to signify if integration should be stopped early (or None for
            all successes)
        """
        raise NotImplementedError()

    def merge_states(
        self, accept: AcceptTensor, current: MethodState, previous: MethodState
    ) -> MethodState:
        raise NotImplementedError()

    def build_interpolation(self, data: InterpolationData) -> LocalInterpolation:
        raise NotImplementedError()

    def convergence_order(self) -> int:
        raise NotImplementedError()
