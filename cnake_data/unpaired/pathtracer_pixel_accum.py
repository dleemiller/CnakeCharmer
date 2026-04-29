from __future__ import annotations


def accumulate_samples(
    rgb_samples: list[list[tuple[float, float, float]]],
) -> list[list[tuple[float, float, float]]]:
    """Average per-pixel RGB samples. Shape: [n_pixels][n_spp]."""
    out: list[list[tuple[float, float, float]]] = []
    for px_samples in rgb_samples:
        if not px_samples:
            out.append([(0.0, 0.0, 0.0)])
            continue
        sr = sg = sb = 0.0
        for r, g, b in px_samples:
            sr += r
            sg += g
            sb += b
        n = float(len(px_samples))
        out.append([(sr / n, sg / n, sb / n)])
    return out
