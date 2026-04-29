"""Elementwise clipped affine combination over two 2D arrays."""

from __future__ import annotations

import numpy as np


def clip_value(a, min_value, max_value):
    return min(max(a, min_value), max_value)


def compute(array_1, array_2, a, b, c):
    array_1 = np.asarray(array_1)
    array_2 = np.asarray(array_2)
    if array_1.shape != array_2.shape:
        raise ValueError("shape mismatch")

    result = np.zeros(
        array_1.shape, dtype=np.result_type(array_1.dtype, array_2.dtype, type(a), type(b), type(c))
    )
    x_max, y_max = array_1.shape

    for x in range(x_max):
        for y in range(y_max):
            tmp = clip_value(array_1[x, y], 2, 10)
            tmp = tmp * a + array_2[x, y] * b
            result[x, y] = tmp + c

    return result
