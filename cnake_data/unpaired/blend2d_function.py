"""2D function blending with clamped mask interpolation."""

from __future__ import annotations


def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


class Blend2D:
    def __init__(self, f1, f2, mask):
        self._f1 = f1
        self._f2 = f2
        self._mask = mask

    def evaluate(self, x, y):
        t = clamp(self._mask(x, y), 0.0, 1.0)
        if t == 0.0:
            return self._f1(x, y)
        if t == 1.0:
            return self._f2(x, y)
        v1 = self._f1(x, y)
        v2 = self._f2(x, y)
        return (1.0 - t) * v1 + t * v2
