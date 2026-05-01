from __future__ import annotations


def rolling_mean(x: list[float], window: int) -> list[float]:
    if window <= 0:
        raise ValueError("window must be positive")
    n = len(x)
    if n == 0:
        return []

    out = [0.0] * n
    acc = 0.0
    for i, v in enumerate(x):
        acc += v
        if i >= window:
            acc -= x[i - window]
        out[i] = acc / min(i + 1, window)
    return out


def normalize_feature(x: list[float]) -> list[float]:
    if not x:
        return []
    lo = min(x)
    hi = max(x)
    if hi == lo:
        return [0.0 for _ in x]
    scale = hi - lo
    return [(v - lo) / scale for v in x]


def make_rul_features(signal: list[float], window: int = 16) -> dict[str, list[float]]:
    m = rolling_mean(signal, window)
    return {
        "raw": signal[:],
        "mean": m,
        "norm_raw": normalize_feature(signal),
        "norm_mean": normalize_feature(m),
    }
