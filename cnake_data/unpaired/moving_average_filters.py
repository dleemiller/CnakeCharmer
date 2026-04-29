"""Simple and exponential moving-average filters."""

from __future__ import annotations


def moving_average_simple(values, n_points):
    out = [0.0] * len(values)
    for i in range(min(n_points, len(values))):
        out[i] = float(values[i])
    for i in range(n_points, len(values)):
        out[i] = sum(values[i - n_points : i]) / float(n_points)
    return out


def moving_average_exp(values, n_points):
    alpha = 2.0 / (n_points + 1)
    d = 0.0
    out = [0.0] * len(values)
    for i, v in enumerate(values):
        out[i] = alpha * v + (1.0 - alpha) * d
        d = out[i]
    return out
