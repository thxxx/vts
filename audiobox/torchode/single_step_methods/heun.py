from torchode.interpolation import ThirdOrderPolynomialInterpolation
from torchode.single_step_methods.runge_kutta import (
    ButcherTableau,
    ERKInterpolationData,
    ExplicitRungeKutta,
)
from torchode.terms import ODETerm


class Heun(ExplicitRungeKutta):
    TABLEAU = ButcherTableau.from_lists(
        c=[0.0, 1.0], a=[[], [1.0]], b=[1 / 2, 1 / 2], b_low_order=[1.0, 0.0]
    )

    def __init__(self, term: ODETerm):
        super().__init__(term, Heun.TABLEAU)

    def convergence_order(self):
        return 2

    def build_interpolation(self, data: ERKInterpolationData):
        return ThirdOrderPolynomialInterpolation.from_k(
            data.t0, data.dt, data.y0, data.y1, data.k
        )
