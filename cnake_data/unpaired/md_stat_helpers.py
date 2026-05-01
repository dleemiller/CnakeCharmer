from __future__ import annotations


def running_mean(x: list[float]) -> list[float]:
    out = [0.0] * len(x)
    s = 0.0
    for i, v in enumerate(x):
        s += v
        out[i] = s / (i + 1)
    return out


def autocorr_lag1(x: list[float]) -> float:
    n = len(x)
    if n < 2:
        return 0.0
    mu = sum(x) / n
    num = 0.0
    den = 0.0
    for i in range(n - 1):
        num += (x[i] - mu) * (x[i + 1] - mu)
    for v in x:
        den += (v - mu) * (v - mu)
    return num / den if den else 0.0
