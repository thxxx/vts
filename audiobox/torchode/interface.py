from collections.abc import Callable

import gradio as gr

from torchode.adjoints import AutoDiffAdjoint
from torchode.problems import InitialValueProblem
from torchode.single_step_methods.runge_kutta import ExplicitRungeKutta
from torchode.solution import Solution
from torchode.step_size_controllers.fixed import FixedStepController
from torchode.step_size_controllers.integral import IntegralController
from torchode.terms import ODETerm
from torchode.typing import DataTensor, EvaluationTimesTensor, TimeTensor

METHODS: dict[str, type[ExplicitRungeKutta]] = {}


def register_method(name: str, constructor: type[ExplicitRungeKutta]):
    METHODS[name] = constructor


def solve_ivp(
    fn: Callable[[TimeTensor, DataTensor], DataTensor],
    y0: DataTensor,
    t_eval: EvaluationTimesTensor,
    *,
    method_class: str | type[ExplicitRungeKutta],
    fixed: bool = True,
    prog: gr.Progress | None = None,
) -> Solution:
    """Solve an initial value problem

    Arguments
    =========
    f
        The dynamics to solve
    y0
        Initial conditions
    t_eval
        Time points to evaluate the solution at
    t_span
        Start and end times of the integration. By default, integrate from the first to
        the last evaluation point.
    method
        Either the name of a registered stepping method, e.g. `"tsit5"`, or a stepping
        method object
    max_steps
        Stop the solver after this many steps
    controller
        Step size controller for the integration. By default a PID controller with
        `atol=1e-7, rtol=1e-7, pcoeff=0.2, icoeff=0.5, dcoeff=0.0` will be
        constructed.
    dt0
        An optional initial time step
    """

    # TODO: Automatically reshape y0 into [batch, features] and back into its original
    # shape

    term = ODETerm(fn)
    if isinstance(method_class, str):
        method = METHODS[method_class](term)
    else:
        method = method_class(term)

    controller = (
        FixedStepController() if fixed else IntegralController(atol=1e-5, rtol=1e-5)
    )
    adjoint = AutoDiffAdjoint(method, controller)
    problem = InitialValueProblem(y0, t_eval)
    return adjoint.solve(problem, term, prog)
