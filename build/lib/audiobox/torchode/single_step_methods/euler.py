from torchode.interpolation import (
    LinearInterpolation,
)
from torchode.single_step_methods.runge_kutta import (
    ButcherTableau,
    ERKInterpolationData,
    ExplicitRungeKutta,
)
from torchode.terms import ODETerm


class Euler(ExplicitRungeKutta):
    # Define the Butcher tableau for Euler's method
    TABLEAU = ButcherTableau.from_lists(
        c=[0.0],  # c value for the single stage
        a=[[]],  # a matrix with no dependency on other stages
        b=[1.0],  # b coefficient for the final step (just k1)
        b_low_order=[1.0],  # Lower order b coefficient, same as b here
    )

    def __init__(self, term: ODETerm):
        # Initialize with the specified Butcher tableau
        super().__init__(term, Euler.TABLEAU)

    def convergence_order(self):
        # Euler's method is of order 1
        return 1

    def build_interpolation(self, data: ERKInterpolationData):
        # Use a third-order polynomial interpolation
        return LinearInterpolation(data.t0, data.dt, data.y0, data.y1)
