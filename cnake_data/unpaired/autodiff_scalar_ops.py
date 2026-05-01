from __future__ import annotations

import math


class AdFloat:
    def __init__(self, x: float, dx: float = 1.0):
        self.x = x
        self.dx = dx

    def __add__(self, b: AdFloat | float) -> AdFloat:
        b = b if isinstance(b, AdFloat) else AdFloat(float(b), 0.0)
        return AdFloat(self.x + b.x, self.dx + b.dx)

    def __sub__(self, b: AdFloat | float) -> AdFloat:
        b = b if isinstance(b, AdFloat) else AdFloat(float(b), 0.0)
        return AdFloat(self.x - b.x, self.dx - b.dx)

    def __mul__(self, b: AdFloat | float) -> AdFloat:
        b = b if isinstance(b, AdFloat) else AdFloat(float(b), 0.0)
        return AdFloat(self.x * b.x, self.dx * b.x + b.dx * self.x)

    def __truediv__(self, b: AdFloat | float) -> AdFloat:
        b = b if isinstance(b, AdFloat) else AdFloat(float(b), 0.0)
        return AdFloat(self.x / b.x, (self.dx * b.x - b.dx * self.x) / (b.x * b.x))


def ad_exp(a: AdFloat | float) -> AdFloat | float:
    if isinstance(a, AdFloat):
        ex = math.exp(a.x)
        return AdFloat(ex, ex * a.dx)
    return math.exp(float(a))
