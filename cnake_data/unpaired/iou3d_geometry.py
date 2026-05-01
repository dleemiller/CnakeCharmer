"""2D rotated-rectangle overlap and 3D IoU helpers."""

from __future__ import annotations

import math


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __sub__(self, v):
        return Point(self.x - v.x, self.y - v.y)

    def cross(self, v):
        return self.x * v.y - self.y * v.x


class Line:
    def __init__(self, v1, v2):
        self.a = v2.y - v1.y
        self.b = v1.x - v2.x
        self.c = v2.cross(v1)

    def __call__(self, p):
        return self.a * p.x + self.b * p.y + self.c

    def intersection(self, other):
        w = self.a * other.b - self.b * other.a
        return Point(
            (self.b * other.c - self.c * other.b) / w, (self.c * other.a - self.a * other.c) / w
        )


def rectangle_vertices_(x1, y1, x2, y2, r):
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    cr = math.cos(r)
    sr = math.sin(r)
    return (
        Point((x1 - cx) * cr + (y1 - cy) * sr + cx, -(x1 - cx) * sr + (y1 - cy) * cr + cy),
        Point((x2 - cx) * cr + (y1 - cy) * sr + cx, -(x2 - cx) * sr + (y1 - cy) * cr + cy),
        Point((x2 - cx) * cr + (y2 - cy) * sr + cx, -(x2 - cx) * sr + (y2 - cy) * cr + cy),
        Point((x1 - cx) * cr + (y2 - cy) * sr + cx, -(x1 - cx) * sr + (y2 - cy) * cr + cy),
    )


def intersection_area(r1, r2):
    rect1 = rectangle_vertices_(*r1)
    rect2 = rectangle_vertices_(*r2)
    inter = list(rect1)
    for p, q in zip(rect2, rect2[1:] + rect2[:1], strict=False):
        if len(inter) <= 2:
            break
        line = Line(p, q)
        new_inter = []
        vals = [line(t) for t in inter]
        for s, t, sv, tv in zip(
            inter, inter[1:] + inter[:1], vals, vals[1:] + vals[:1], strict=False
        ):
            if sv <= 0:
                new_inter.append(s)
            if sv * tv < 0:
                new_inter.append(line.intersection(Line(s, t)))
        inter = new_inter
    if len(inter) <= 2:
        return 0.0
    return 0.5 * sum(
        p.x * q.y - p.y * q.x for p, q in zip(inter, inter[1:] + inter[:1], strict=False)
    )
