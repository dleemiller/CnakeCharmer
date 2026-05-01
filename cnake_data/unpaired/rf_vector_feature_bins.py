from __future__ import annotations


def feature_histogram(values: list[float], lo: float, hi: float, n_bins: int) -> list[int]:
    if n_bins <= 0 or hi <= lo:
        raise ValueError("invalid bin config")
    bins = [0] * n_bins
    span = hi - lo
    for v in values:
        t = (v - lo) / span
        idx = int(t * n_bins)
        if idx < 0:
            idx = 0
        elif idx >= n_bins:
            idx = n_bins - 1
        bins[idx] += 1
    return bins


def normalized_histogram(values: list[float], lo: float, hi: float, n_bins: int) -> list[float]:
    b = feature_histogram(values, lo, hi, n_bins)
    n = len(values)
    if n == 0:
        return [0.0] * n_bins
    return [x / n for x in b]
