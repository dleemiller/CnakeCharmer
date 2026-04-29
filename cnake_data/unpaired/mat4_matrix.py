"""Small 4x4 matrix helper routines."""

from __future__ import annotations


def identity4():
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def as_mat4(values):
    """Coerce row-major 16 values or nested 4x4 values into a 4x4 matrix."""
    if len(values) == 4 and all(hasattr(row, "__len__") and len(row) == 4 for row in values):
        return [[float(values[i][j]) for j in range(4)] for i in range(4)]

    if len(values) == 16:
        return [[float(values[i * 4 + j]) for j in range(4)] for i in range(4)]

    raise ValueError("expected 4x4 nested values or 16 row-major values")


def mat4_mul(a, b):
    out = [[0.0] * 4 for _ in range(4)]
    for i in range(4):
        for j in range(4):
            total = 0.0
            for k in range(4):
                total += a[i][k] * b[k][j]
            out[i][j] = total
    return out
