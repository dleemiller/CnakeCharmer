"""2D affine transformation helpers and lightweight class wrapper."""

from __future__ import annotations

import math
from dataclasses import dataclass


def from_flat(data):
    a, b, c, d, x, y = data
    return Affine2(a, b, c, d, x, y)


def from_rows(rows):
    (a, b, x), (c, d, y) = rows
    return Affine2(a, b, c, d, x, y)


@dataclass
class Affine2:
    a: float
    b: float
    c: float
    d: float
    x: float
    y: float

    def transform_point(self, x: float, y: float):
        return (self.a * x + self.b * y + self.x, self.c * x + self.d * y + self.y)

    def translate_xy(self, dx: float, dy: float):
        return Affine2(self.a, self.b, self.c, self.d, self.x + dx, self.y + dy)

    def __call__(self, v):
        return self.transform_point(v[0], v[1])

    def is_null(self):
        return (
            self.a == 0
            and self.b == 0
            and self.c == 0
            and self.d == 0
            and self.x == 0
            and self.y == 0
        )

    def angle_rad(self):
        return math.atan2(self.c, self.a)
