from typing import Protocol, runtime_checkable

import torch
from einops import einsum

from torchode.typing import (
    DataTensor,
    InterpDataTensor,
    InterpTimeTensor,
    SampleIndexTensor,
    StepTensor,
    TimeTensor,
    WeightVector,
)


def poly3eval(
    d: InterpDataTensor,
    c: InterpDataTensor,
    b: InterpDataTensor,
    a: InterpDataTensor,
    t: InterpTimeTensor,
    t0: InterpTimeTensor,
    t1: InterpTimeTensor,
):
    """Evaluate a 3rd order polynomial on the interval [t0, t1].

    The coefficients a..3 define the polynomial on the interval [0, 1].
    """
    dt = t1 - t0
    dt = torch.where(dt == 0, 1.0, dt)
    x = ((t - t0) / dt).to(dtype=a.dtype).reshape(-1, *([1] * (a.ndim - 1)))

    # Evaluate the polynomial with Horner's method
    y = a
    y = torch.addcmul(b, y, x)
    y = torch.addcmul(c, y, x)
    y = torch.addcmul(d, y, x)
    return y


def poly4eval(
    e: InterpDataTensor,
    d: InterpDataTensor,
    c: InterpDataTensor,
    b: InterpDataTensor,
    a: InterpDataTensor,
    t: InterpTimeTensor,
    t0: InterpTimeTensor,
    t1: InterpTimeTensor,
):
    """Evaluate a 4th order polynomial on the interval [t0, t1].

    The coefficients a..e define the polynomial on the interval [0, 1].
    """
    dt = t1 - t0
    dt = torch.where(dt == 0, 1.0, dt)
    x = ((t - t0) / (dt)).to(dtype=a.dtype).reshape(-1, *([1] * (a.ndim - 1)))

    # Evaluate the polynomial with Horner's method
    y = a
    y = torch.addcmul(b, y, x)
    y = torch.addcmul(c, y, x)
    y = torch.addcmul(d, y, x)
    y = torch.addcmul(e, y, x)
    return y


@runtime_checkable
class LocalInterpolation(Protocol):
    def evaluate(self, t: TimeTensor, idx: SampleIndexTensor) -> DataTensor:
        raise NotImplementedError()


class LinearInterpolation(LocalInterpolation):
    def __init__(self, t0: TimeTensor, dt: TimeTensor, y0: DataTensor, y1: DataTensor):
        self.t0 = t0
        self.dt = dt
        self.y0 = y0
        self.dy = y1 - y0

    def evaluate(self, t: TimeTensor, idx: SampleIndexTensor) -> DataTensor:
        dt = self.dt[idx]
        dt = torch.where(dt == 0, 1.0, dt)
        x = (
            ((t - self.t0[idx]) / dt)
            .to(dtype=self.y0.dtype)
            .reshape(-1, *([1] * (self.y0.ndim - 1)))
        )

        return torch.addcmul(self.y0[idx], self.dy[idx], x)


class ThirdOrderPolynomialInterpolation(LocalInterpolation):
    """Third-order polynomial interpolation on [t0, t1].

    `coefficients` holds the coefficients of a third-order polynomial on [0, 1] in
    increasing order, i.e. `cofficients[i]` belongs to `x**i`.
    """

    def __init__(
        self,
        t0: TimeTensor,
        t1: TimeTensor,
        coefficients: tuple[DataTensor, DataTensor, DataTensor, DataTensor],
    ):
        self.t0 = t0
        self.t1 = t1
        self.coefficients = coefficients

    @staticmethod
    def from_k(
        t0: TimeTensor,
        dt: TimeTensor,
        y0: DataTensor,
        y1: DataTensor,
        k: StepTensor,
    ):
        """Find the coefficients from the k values of a Runge-Kutta step

        Computes the coefficients by fitting a third-order Hermite polynomial and then
        translating the found coefficients into the simple x**p polynomial form.
        """

        f0 = k[0]
        f1 = k[-1]
        dt_data = dt.to(dtype=y0.dtype)

        # The transformation from Hermite to normal coefficients is already included in
        # these formulas
        y0my1 = y0 - y1
        a = torch.add(einsum(dt_data, (f0 + f1), "b, b ... -> b ..."), y0my1, alpha=2)
        b = torch.add(
            einsum(dt_data, torch.add(-f1, f0, alpha=-2), "b, b ... -> b ..."),
            y0my1,
            alpha=-3,
        )
        c = einsum(dt_data, f0, "b, b ... -> b ...")
        d = y0

        coefficients = (d, c, b, a)
        return ThirdOrderPolynomialInterpolation(t0, t0 + dt, coefficients)

    def evaluate(self, t: InterpTimeTensor, idx: SampleIndexTensor) -> InterpDataTensor:
        d, c, b, a = self.coefficients
        coeff = (d[idx], c[idx], b[idx], a[idx])
        return poly3eval(*coeff, t, self.t0[idx], self.t1[idx])

    def __repr__(self):
        return (
            f"ThirdOrderPolynomialInterpolation(t0={self.t0}, t1={self.t1}, "
            f"coefficients={self.coefficients})"
        )


class FourthOrderPolynomialInterpolation(LocalInterpolation):
    """Polynomial interpolation on [t0, t1].

    `coefficients` holds the coefficients of a fourth-order polynomial on [0, 1] in
    increasing order, i.e. `cofficients[i]` belongs to `x**i`.
    """

    def __init__(
        self,
        t0: TimeTensor,
        t1: TimeTensor,
        coefficients: tuple[DataTensor, DataTensor, DataTensor, DataTensor, DataTensor],
    ):
        self.t0 = t0
        self.t1 = t1
        self.coefficients = coefficients

    @staticmethod
    def from_k(
        t0: TimeTensor,
        dt: TimeTensor,
        y0: DataTensor,
        y1: DataTensor,
        k: StepTensor,
        b_mid: WeightVector,
    ):
        dt_data = dt.to(dtype=y0.dtype)
        f0 = einsum(dt_data, k[0], "b, b ... -> b ...")
        f1 = einsum(dt_data, k[-1], "b, b ... -> b ...")
        y_mid = y0 + einsum(dt_data, b_mid, k, "b, w, w b ... -> b ...")

        a = (2 * (f1 - f0)).add(y1 + y0, alpha=-8).add(y_mid, alpha=16)
        b = (
            (5 * f0)
            .add(f1, alpha=-3)
            .add(y0, alpha=18)
            .add(y1, alpha=14)
            .add(y_mid, alpha=-32)
        )
        c = (
            f1.add(f0, alpha=-4)
            .add(y0, alpha=-11)
            .add(y1, alpha=-5)
            .add(y_mid, alpha=16)
        )
        d = f0
        e = y0

        coefficients = (e, d, c, b, a)
        return FourthOrderPolynomialInterpolation(t0, t0 + dt, coefficients)

    def evaluate(self, t: InterpTimeTensor, idx: SampleIndexTensor) -> InterpDataTensor:
        e, d, c, b, a = self.coefficients
        coeff = (e[idx], d[idx], c[idx], b[idx], a[idx])
        return poly4eval(*coeff, t, self.t0[idx], self.t1[idx])

    def __repr__(self):
        return (
            f"FourthOrderPolynomialInterpolation(t0={self.t0}, t1={self.t1}, "
            f"coefficients={self.coefficients})"
        )
