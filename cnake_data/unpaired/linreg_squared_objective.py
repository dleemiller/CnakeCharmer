from __future__ import annotations


def linreg_squared_error(x: list[float], y: list[float], a: float, b: float) -> float:
    if len(x) != len(y):
        raise ValueError("x and y must have same length")
    s = 0.0
    for i in range(len(x)):
        e = (a * x[i] + b) - y[i]
        s += e * e
    return s


def linreg_gradients(x: list[float], y: list[float], a: float, b: float) -> tuple[float, float]:
    if len(x) != len(y):
        raise ValueError("x and y must have same length")
    da = 0.0
    db = 0.0
    for i in range(len(x)):
        e = (a * x[i] + b) - y[i]
        da += 2.0 * e * x[i]
        db += 2.0 * e
    return da, db
