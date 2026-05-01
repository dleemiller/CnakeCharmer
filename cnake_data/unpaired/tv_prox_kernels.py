from __future__ import annotations


def prox_tv1d_simple(w: list[float], stepsize: float) -> list[float]:
    if not w:
        return []
    x = w[:]
    for i in range(1, len(x) - 1):
        left = x[i] - x[i - 1]
        right = x[i + 1] - x[i]
        if left > stepsize:
            x[i] -= stepsize
        elif left < -stepsize:
            x[i] += stepsize
        if right > stepsize:
            x[i] += stepsize * 0.5
        elif right < -stepsize:
            x[i] -= stepsize * 0.5
    return x


def prox_tv2d_rows_cols(
    x: list[list[float]], stepsize: float, iters: int = 10
) -> list[list[float]]:
    out = [row[:] for row in x]
    n_rows = len(out)
    n_cols = len(out[0]) if n_rows else 0
    for _ in range(iters):
        for r in range(n_rows):
            out[r] = prox_tv1d_simple(out[r], stepsize)
        for c in range(n_cols):
            col = [out[r][c] for r in range(n_rows)]
            col = prox_tv1d_simple(col, stepsize)
            for r in range(n_rows):
                out[r][c] = col[r]
    return out
