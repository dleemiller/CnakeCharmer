"""Dense correlation entry, Jacobian, and Hessian wrt anisotropic scale."""

from __future__ import annotations

import math
from collections.abc import Callable


def euclidean_distance(point1: list[float], point2: list[float], scale: list[float]) -> float:
    total = 0.0
    for a, b, s in zip(point1, point2, scale, strict=False):
        d = (a - b) / s
        total += d * d
    return math.sqrt(total)


def compute_dense_correlation(
    point1: list[float],
    point2: list[float],
    scale: list[float],
    kernel: Callable[[float], float],
) -> float:
    distance = euclidean_distance(point1, point2, scale)
    return kernel(distance)


def compute_dense_correlation_jacobian(
    points: list[list[float]],
    scale: list[float],
    kernel_d1: Callable[[float], float],
    i: int,
    j: int,
) -> list[float]:
    dim = len(scale)
    if i == j:
        return [0.0] * dim

    distance = euclidean_distance(points[i], points[j], scale)
    d1_kernel = kernel_d1(distance)

    out = [0.0] * dim
    for p in range(dim):
        delta = points[i][p] - points[j][p]
        d1_distance = -(delta * delta) / (distance * (scale[p] ** 3))
        out[p] = d1_kernel * d1_distance

    return out


def compute_dense_correlation_hessian(
    points: list[list[float]],
    scale: list[float],
    kernel_d1: Callable[[float], float],
    kernel_d2: Callable[[float], float],
    i: int,
    j: int,
) -> list[list[float]]:
    dim = len(scale)
    out = [[0.0 for _ in range(dim)] for _ in range(dim)]
    if i == j:
        return out

    distance = euclidean_distance(points[i], points[j], scale)
    d1_kernel = kernel_d1(distance)
    d2_kernel = kernel_d2(distance)

    for p in range(dim):
        for q in range(p, dim):
            dp = points[i][p] - points[j][p]
            dq = points[i][q] - points[j][q]

            dp_distance = -(dp * dp) / (distance * (scale[p] ** 3))
            if q == p:
                dq_distance = dp_distance
                dpq_distance = ((dp * dp) / (distance * (scale[p] ** 3))) * (
                    3.0 / scale[p] + dp_distance / distance
                )
            else:
                dq_distance = -(dq * dq) / (distance * (scale[q] ** 3))
                dpq_distance = ((dp * dp) / (distance**2 * (scale[p] ** 3))) * dq_distance

            value = d2_kernel * dp_distance * dq_distance + d1_kernel * dpq_distance
            out[p][q] = value
            out[q][p] = value

    return out
