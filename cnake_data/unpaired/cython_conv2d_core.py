from __future__ import annotations


def conv2d_valid(img: list[list[float]], ker: list[list[float]]) -> list[list[float]]:
    h = len(img)
    w = len(img[0]) if h else 0
    kh = len(ker)
    kw = len(ker[0]) if kh else 0
    oh = h - kh + 1
    ow = w - kw + 1
    if oh <= 0 or ow <= 0:
        return []
    out = [[0.0] * ow for _ in range(oh)]
    for y in range(oh):
        for x in range(ow):
            s = 0.0
            for ky in range(kh):
                for kx in range(kw):
                    s += img[y + ky][x + kx] * ker[ky][kx]
            out[y][x] = s
    return out
