from torchode.interpolation import ThirdOrderPolynomialInterpolation
from torchode.single_step_methods.runge_kutta import (
    ButcherTableau,
    ERKInterpolationData,
    ExplicitRungeKutta,
)
from torchode.terms import ODETerm


class Midpoint(ExplicitRungeKutta):
    # Define the Butcher tableau for the Midpoint method
    TABLEAU = ButcherTableau.from_lists(
        c=[0.0, 0.5],  # c values
        a=[[], [0.5]],  # a matrix
        b=[0.0, 1.0],  # b coefficients for final step
        b_low_order=[1.0, 0.0],  # Lower order b coefficients (Euler fallback)
    )

    def __init__(self, term: ODETerm):
        # Initialize with the specified Butcher tableau
        super().__init__(term, Midpoint.TABLEAU)

    def convergence_order(self):
        # Midpoint method is of order 2
        return 2

    def build_interpolation(self, data: ERKInterpolationData):
        # Use a third-order polynomial interpolation
        return ThirdOrderPolynomialInterpolation.from_k(
            data.t0, data.dt, data.y0, data.y1, data.k
        )
