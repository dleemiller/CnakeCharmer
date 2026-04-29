"""Batch vector rotation using axis-angle and quaternions."""

from __future__ import annotations

import math


def _quat_from_angle_axis(angle, axis):
    ax, ay, az = axis
    norm = math.sqrt(ax * ax + ay * ay + az * az)
    if norm == 0:
        return (1.0, 0.0, 0.0, 0.0)
    ax, ay, az = ax / norm, ay / norm, az / norm
    s = math.sin(angle / 2.0)
    return (math.cos(angle / 2.0), ax * s, ay * s, az * s)


def _quat_rotate(v, q):
    w, x, y, z = q
    vx, vy, vz = v
    tx = 2 * (y * vz - z * vy)
    ty = 2 * (z * vx - x * vz)
    tz = 2 * (x * vy - y * vx)
    return (
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    )


def rotate(vectors, axes, angles):
    out = [[0.0, 0.0, 0.0] for _ in range(len(vectors))]
    n_axes = len(axes)
    n_angles = len(angles)
    for row in range(len(vectors)):
        q = _quat_from_angle_axis(angles[row % n_angles], axes[row % n_axes])
        rx, ry, rz = _quat_rotate(vectors[row], q)
        out[row][0], out[row][1], out[row][2] = rx, ry, rz
    return out
