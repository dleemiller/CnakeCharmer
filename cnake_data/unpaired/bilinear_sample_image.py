from __future__ import annotations


def _mirror_index(val: int, lower: int, upper: int) -> int:
    while True:
        valid = True
        if val >= upper:
            val = 2 * upper - val - 1
            valid = False
        if val < lower:
            val = -val
            valid = False
        if valid:
            return val


def bilinear_sample(
    img: list[list[list[int]]], x: float, y: float, mirror: bool = False
) -> list[float]:
    h = len(img)
    w = len(img[0]) if h else 0
    ccount = len(img[0][0]) if w else 0

    xi = int(x)
    yi = int(y)
    xi2 = xi + 1
    yi2 = yi + 1
    xfrac = x - xi
    yfrac = y - yi

    if mirror:
        xi = _mirror_index(xi, 0, w)
        xi2 = _mirror_index(xi2, 0, w)
        yi = _mirror_index(yi, 0, h)
        yi2 = _mirror_index(yi2, 0, h)

    out = [0.0] * ccount
    for ch in range(ccount):
        p00 = img[yi][xi][ch]
        p10 = img[yi][xi2][ch]
        p01 = img[yi2][xi][ch]
        p11 = img[yi2][xi2][ch]
        c1 = p00 * (1.0 - xfrac) + p10 * xfrac
        c2 = p01 * (1.0 - xfrac) + p11 * xfrac
        out[ch] = c1 * (1.0 - yfrac) + c2 * yfrac
    return out
