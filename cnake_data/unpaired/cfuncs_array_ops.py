from __future__ import annotations


def saxpy(a: float, x: list[float], y: list[float]) -> list[float]:
    if len(x) != len(y):
        raise ValueError("length mismatch")
    out = [0.0] * len(x)
    for i in range(len(x)):
        out[i] = a * x[i] + y[i]
    return out


def dot(x: list[float], y: list[float]) -> float:
    if len(x) != len(y):
        raise ValueError("length mismatch")
    s = 0.0
    for i in range(len(x)):
        s += x[i] * y[i]
    return s
