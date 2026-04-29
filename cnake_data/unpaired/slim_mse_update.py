from __future__ import annotations


def dot(a: list[float], b: list[float]) -> float:
    s = 0.0
    for i in range(len(a)):
        s += a[i] * b[i]
    return s


def sgd_weight_step(w: list[float], x: list[float], y: float, lr: float, l2: float) -> list[float]:
    pred = dot(w, x)
    err = pred - y
    out = w[:]
    for i in range(len(w)):
        grad = err * x[i] + l2 * w[i]
        out[i] -= lr * grad
    return out
