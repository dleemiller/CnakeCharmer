from __future__ import annotations


def estimate_sigma(values: list[list[float]]) -> float:
    n = len(values)
    if n == 0:
        return 0.0
    sigma = 0.0
    for obs in values:
        m = len(obs)
        if m < 2:
            continue
        s = 0.0
        for k in range(1, m):
            d = obs[k] - obs[k - 1]
            s += d * d
        sigma += s / (2.0 * (m - 1))
    return (sigma / n) ** 0.5


def estimate_sigma_mse(values: list[list[float]], values_estim: list[list[float]]) -> float:
    n = len(values)
    if n == 0:
        return 0.0
    sigma = 0.0
    for obs, est in zip(values, values_estim, strict=False):
        m = len(obs)
        if m == 0:
            continue
        s = 0.0
        for k in range(m):
            d = obs[k] - est[k]
            s += d * d
        sigma += s / m
    return (sigma / n) ** 2
