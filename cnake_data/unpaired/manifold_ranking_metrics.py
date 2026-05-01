"""Trustworthiness, continuity, and LCMC metrics from co-ranking matrices."""

from __future__ import annotations


def _tc_normalisation_weight(n_samples: int, k: int) -> float:
    if k <= 0 or k >= n_samples:
        return 1.0
    return 2.0 / (n_samples * k * (2.0 * n_samples - 3.0 * k - 1.0))


def trustworthiness(q: list[list[int]], k: int) -> float:
    n = len(q)
    if n == 0 or k <= 0:
        return 0.0

    intrusions = 0.0
    for i in range(k, n):
        for j in range(k):
            intrusions += (i - k + 1) * q[i][j]

    return 1.0 - _tc_normalisation_weight(n, k) * intrusions


def continuity(q: list[list[int]], k: int) -> float:
    n = len(q)
    if n == 0 or k <= 0:
        return 0.0

    extrusions = 0.0
    for i in range(k):
        for j in range(k, n):
            extrusions += (j - k + 1) * q[i][j]

    return 1.0 - _tc_normalisation_weight(n, k) * extrusions


def lcmc(q: list[list[int]], k: int) -> float:
    n = len(q)
    if n == 0 or k <= 0:
        return 0.0

    local_overlap = 0.0
    for i in range(k):
        for j in range(k):
            local_overlap += q[i][j]

    return (local_overlap / (n * k)) - (k / (n - 1.0))
