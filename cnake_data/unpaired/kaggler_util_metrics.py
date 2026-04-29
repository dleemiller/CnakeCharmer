from __future__ import annotations


def weighted_mean(values: list[float], weights: list[float]) -> float:
    if len(values) != len(weights):
        raise ValueError("length mismatch")
    sw = 0.0
    sv = 0.0
    for v, w in zip(values, weights, strict=False):
        sv += v * w
        sw += w
    return sv / sw if sw != 0.0 else 0.0


def gini_coefficient(values: list[float]) -> float:
    n = len(values)
    if n == 0:
        return 0.0
    s = sorted(values)
    cum = 0.0
    tot = sum(s)
    if tot == 0.0:
        return 0.0
    for i, x in enumerate(s, start=1):
        cum += i * x
    return (2.0 * cum) / (n * tot) - (n + 1.0) / n
