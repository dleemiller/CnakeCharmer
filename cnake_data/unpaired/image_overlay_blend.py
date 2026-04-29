from __future__ import annotations


def alpha_blend_rgba(
    base: list[list[tuple[int, int, int, int]]],
    over: list[list[tuple[int, int, int, int]]],
) -> list[list[tuple[int, int, int, int]]]:
    h = len(base)
    if h == 0:
        return []
    w = len(base[0])
    out = [[(0, 0, 0, 0)] * w for _ in range(h)]

    for y in range(h):
        for x in range(w):
            br, bg, bb, ba = base[y][x]
            or_, og, ob, oa = over[y][x]
            a_o = oa / 255.0
            a_b = ba / 255.0
            a = a_o + a_b * (1.0 - a_o)
            if a <= 1e-12:
                out[y][x] = (0, 0, 0, 0)
                continue
            r = (or_ * a_o + br * a_b * (1.0 - a_o)) / a
            g = (og * a_o + bg * a_b * (1.0 - a_o)) / a
            b = (ob * a_o + bb * a_b * (1.0 - a_o)) / a
            out[y][x] = (int(r), int(g), int(b), int(a * 255.0))
    return out
