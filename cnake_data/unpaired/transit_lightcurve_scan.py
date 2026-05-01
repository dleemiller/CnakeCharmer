from __future__ import annotations


def rebin_sum(arr: list[list[float]], out_size: int) -> list[list[float]]:
    n = len(arr)
    if out_size <= 0 or n == 0:
        return []
    factor = n // out_size
    out = [[0.0 for _ in range(out_size)] for _ in range(out_size)]
    for oy in range(out_size):
        for ox in range(out_size):
            s = 0.0
            for iy in range(factor):
                for ix in range(factor):
                    s += arr[oy * factor + iy][ox * factor + ix]
            out[oy][ox] = s
    m = max(max(r) for r in out) if out else 1.0
    if m > 0:
        for y in range(out_size):
            for x in range(out_size):
                out[y][x] /= m
    return out


def circular_mask(radius: float, supersample_factor: int) -> list[list[float]]:
    r_up = int((radius + 1) // 1)
    n = 2 * r_up * supersample_factor
    mask = [[0.0 for _ in range(n)] for _ in range(n)]
    rr = (radius * supersample_factor) * (radius * supersample_factor)
    c = r_up * supersample_factor
    for i in range(n):
        for j in range(n):
            d = (i - c) * (i - c) + (j - c) * (j - c)
            if d < rr:
                mask[i][j] = 1.0
    return rebin_sum(mask, 2 * r_up)
