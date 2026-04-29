from __future__ import annotations


def radial_histogram(r: list[float], bins: list[float]) -> list[int]:
    if len(bins) < 2:
        return []
    out = [0] * (len(bins) - 1)
    for x in r:
        for i in range(len(out)):
            if bins[i] <= x < bins[i + 1]:
                out[i] += 1
                break
    return out


def normalize_hist(hist: list[int]) -> list[float]:
    s = sum(hist)
    if s == 0:
        return [0.0] * len(hist)
    return [v / s for v in hist]
