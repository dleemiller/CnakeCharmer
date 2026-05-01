"""Discrete impulse/heaviside/ramp linear combinations."""

from __future__ import annotations

import numpy as np


def discrete_impulse_sum(sampled_n, k_array, a_array):
    result = np.zeros(sampled_n.size, dtype=float)
    for a, k in zip(a_array, k_array, strict=False):
        impulse = np.zeros(sampled_n.size, dtype=float)
        if 0 <= k < sampled_n.size:
            impulse[k] = 1.0
        result += a * impulse
    return result


def discrete_heaviside_sum(sampled_n, k_array, a_array):
    result = np.zeros(sampled_n.size, dtype=float)
    for a, k in zip(a_array, k_array, strict=False):
        result += a * np.heaviside(np.subtract(sampled_n, k), 1)
    return result


def discrete_ramp_sum(sampled_n, k_array, a_array):
    result = np.zeros(sampled_n.size, dtype=float)
    for a, k in zip(a_array, k_array, strict=False):
        result += a * np.subtract(sampled_n, k).clip(0)
    return result
