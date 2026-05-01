"""Uniform and truncated-normal PDF/CDF primitives."""

from __future__ import annotations

import math


def standard_normal_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def standard_normal_pdf(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def uniform_cdf(x, a, b):
    if x < a:
        return 0.0
    if x > b:
        return 1.0
    return (x - a) / (b - a)


def uniform_pdf(x, a, b):
    if x < a or x > b:
        return 0.0
    return 1.0 / (b - a)


def truncated_normal_cdf(x, loc, scale, a, b):
    eps = (x - loc) / scale
    alpha = (a - loc) / scale
    beta = (b - loc) / scale
    if x < a:
        return 0.0
    if x > b:
        return 1.0
    return (standard_normal_cdf(eps) - standard_normal_cdf(alpha)) / (
        standard_normal_cdf(beta) - standard_normal_cdf(alpha)
    )


def truncated_normal_pdf(x, loc, scale, a, b):
    eps = (x - loc) / scale
    alpha = (a - loc) / scale
    beta = (b - loc) / scale
    if x < a or x > b:
        return 0.0
    return standard_normal_pdf(eps) / (
        scale * (standard_normal_cdf(beta) - standard_normal_cdf(alpha))
    )
