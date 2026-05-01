from __future__ import annotations

import math


def buffon_hits(
    angles: list[float], distances: list[float], needle_len: float, line_gap: float
) -> int:
    """Count crossings in Buffon's needle setup for pre-sampled angle/distance pairs."""
    if len(angles) != len(distances):
        raise ValueError("angles/distances mismatch")
    hits = 0
    half = needle_len * 0.5
    for a, d in zip(angles, distances, strict=False):
        if d <= half * abs(math.sin(a)):
            hits += 1
    return hits


def estimate_pi(n_trials: int, hits: int, needle_len: float, line_gap: float) -> float:
    if hits <= 0 or line_gap <= 0 or needle_len <= 0 or n_trials <= 0:
        return float("inf")
    return (2.0 * needle_len * n_trials) / (line_gap * hits)
