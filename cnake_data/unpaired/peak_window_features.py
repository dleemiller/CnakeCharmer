from __future__ import annotations


def local_maxima(x: list[float]) -> list[int]:
    out: list[int] = []
    for i in range(1, len(x) - 1):
        if x[i] >= x[i - 1] and x[i] >= x[i + 1]:
            out.append(i)
    return out


def window_sum(x: list[float], center: int, radius: int) -> float:
    lo = max(0, center - radius)
    hi = min(len(x), center + radius + 1)
    s = 0.0
    for i in range(lo, hi):
        s += x[i]
    return s


def peak_scores(x: list[float], radius: int = 2) -> list[tuple[int, float]]:
    return [(i, window_sum(x, i, radius)) for i in local_maxima(x)]
