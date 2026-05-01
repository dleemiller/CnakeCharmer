from __future__ import annotations


def radial_intensity_profile(
    x: list[float],
    y: list[float],
    intensity: list[float],
    cx: float,
    cy: float,
    r_bins: list[float],
) -> list[float]:
    if not (len(x) == len(y) == len(intensity)):
        raise ValueError("length mismatch")
    if len(r_bins) < 2:
        return []

    sums = [0.0] * (len(r_bins) - 1)
    counts = [0] * (len(r_bins) - 1)
    for i in range(len(x)):
        dx = x[i] - cx
        dy = y[i] - cy
        r = (dx * dx + dy * dy) ** 0.5
        for b in range(len(r_bins) - 1):
            if r_bins[b] <= r < r_bins[b + 1]:
                sums[b] += intensity[i]
                counts[b] += 1
                break

    out = [0.0] * len(sums)
    for i in range(len(out)):
        out[i] = sums[i] / counts[i] if counts[i] else 0.0
    return out
