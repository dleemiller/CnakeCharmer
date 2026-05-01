"""2D point utilities with rectangular/polar conversions and axis rotation."""

from __future__ import annotations

import math
from dataclasses import dataclass


def polar2rect(radius: float, angle: float) -> tuple[float, float]:
    return radius * math.cos(angle), radius * math.sin(angle)


def rect2polar(x: float, y: float) -> tuple[float, float]:
    return math.hypot(x, y), math.atan2(y, x)


def rotate_point(
    x: float, y: float, axis_x: float, axis_y: float, angle: float
) -> tuple[float, float]:
    dx0 = x - axis_x
    dy0 = y - axis_y
    d0 = math.hypot(dx0, dy0)
    w0 = math.atan2(dy0, dx0)
    dx1 = d0 * math.cos(w0 + angle)
    dy1 = d0 * math.sin(w0 + angle)
    return axis_x + dx1, axis_y + dy1


@dataclass
class Point2D:
    x: float
    y: float

    @property
    def radius(self) -> float:
        return math.hypot(self.x, self.y)

    @property
    def angle(self) -> float:
        return math.atan2(self.y, self.x)

    def rotate(self, angle: float, axis_x: float = 0.0, axis_y: float = 0.0) -> Point2D:
        rx, ry = rotate_point(self.x, self.y, axis_x, axis_y, angle)
        return Point2D(rx, ry)
