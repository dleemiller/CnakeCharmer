from __future__ import annotations


def interp1d(xs: list[float], ys: list[float], xq: list[float]) -> list[float]:
    if len(xs) != len(ys):
        raise ValueError("xs/ys mismatch")
    if len(xs) < 2:
        return [ys[0] if ys else 0.0 for _ in xq]

    out: list[float] = []
    j = 0
    for x in xq:
        while j + 1 < len(xs) - 1 and xs[j + 1] < x:
            j += 1
        x0, x1 = xs[j], xs[j + 1]
        y0, y1 = ys[j], ys[j + 1]
        if x1 == x0:
            out.append(y0)
        else:
            t = (x - x0) / (x1 - x0)
            out.append(y0 + t * (y1 - y0))
    return out
