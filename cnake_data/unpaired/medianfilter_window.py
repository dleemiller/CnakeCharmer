from __future__ import annotations


def median1d(x: list[float], radius: int) -> list[float]:
    if radius < 0:
        raise ValueError("radius must be >= 0")
    n = len(x)
    out = [0.0] * n
    for i in range(n):
        lo = max(0, i - radius)
        hi = min(n, i + radius + 1)
        w = sorted(x[lo:hi])
        out[i] = w[len(w) // 2]
    return out
