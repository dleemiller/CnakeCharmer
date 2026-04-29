"""Node geometry helpers for 2D/3D operations."""

from __future__ import annotations

import math

import numpy as np


def _as_2d(a):
    return np.array([a[0], a[1], 0.0])


def _as_unit_vector(a):
    return a / np.linalg.norm(a)


def _dist_3d(a, b):
    return math.sqrt((b[0] - a[0]) ** 2.0 + (b[1] - a[1]) ** 2.0 + (b[2] - a[2]) ** 2.0)


def _dist_2d(a, b):
    return math.sqrt((b[0] - a[0]) ** 2.0 + (b[1] - a[1]) ** 2.0)


def _is_point_btw(a, b, point):
    d_apb = _dist_3d(a, point) + _dist_3d(b, point)
    d_ab = _dist_3d(a, b)
    return abs(d_ab - d_apb) < 0.1


def _eq(a, b):
    return a[0] == b[0] and a[1] == b[1] and a[2] == b[2]


class Node:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.data = np.array((float(x), float(y), float(z)))

    def x(self, x=math.nan):
        if not math.isnan(x):
            self.data[0] = x
        return self.data[0]

    def y(self, y=math.nan):
        if not math.isnan(y):
            self.data[1] = y
        return self.data[1]

    def z(self, z=math.nan):
        if not math.isnan(z):
            self.data[2] = z
        return self.data[2]

    def as_ndarray(self):
        return np.array([self.x(), self.y(), self.z()])

    def as_ndarray_2d(self):
        return np.array([self.x(), self.y()])

    def as_list_2d(self):
        return [self.x(), self.y()]

    def as_list_3d(self):
        return [self.x(), self.y(), self.z()]

    def as_node_2d(self):
        return Node(self.x(), self.y(), 0)

    def as_unit_vector(self):
        return Node.from_array(self.as_ndarray() / np.linalg.norm(self.as_ndarray()))

    def mag(self):
        return np.linalg.norm(self.as_ndarray())

    @classmethod
    def from_array(cls, arr):
        arr = np.asarray(arr)
        if arr.shape[0] == 2:
            return cls(arr[0], arr[1], 0.0)
        return cls(arr[0], arr[1], arr[2])

    @classmethod
    def from_tuple(cls, t):
        return cls.from_array(np.array(t, dtype=float))

    @classmethod
    def from_list(cls, t):
        return cls.from_array(np.array(t, dtype=float))
