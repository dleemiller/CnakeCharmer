"""Quaternion and homogeneous transform utilities."""

from __future__ import annotations

import math

import numpy as np

_EPS = np.finfo(float).eps * 4.0


def quaternion_from_matrix(M: np.ndarray) -> np.ndarray:
    q = np.empty((4,), dtype=np.float64)
    t = float(np.trace(M))
    if t > M[3, 3]:
        q[3] = t
        q[2] = M[1, 0] - M[0, 1]
        q[1] = M[0, 2] - M[2, 0]
        q[0] = M[2, 1] - M[1, 2]
    else:
        i, j, k = 0, 1, 2
        if M[1, 1] > M[0, 0]:
            i, j, k = 1, 2, 0
        if M[2, 2] > M[i, i]:
            i, j, k = 2, 0, 1
        t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
        q[i] = t
        q[j] = M[i, j] + M[j, i]
        q[k] = M[k, i] + M[i, k]
        q[3] = M[k, j] - M[j, k]
    q *= 0.5 / math.sqrt(t * M[3, 3])
    return q


def quaternion_matrix(iq: np.ndarray) -> np.ndarray:
    iq = np.array(iq[:4], dtype=np.float64, copy=True)
    nq = float(np.dot(iq, iq))
    if nq < _EPS:
        return np.identity(4)
    iq *= math.sqrt(2.0 / nq)
    q = np.outer(iq, iq)
    return np.array(
        (
            (1.0 - q[1, 1] - q[2, 2], q[0, 1] - q[2, 3], q[0, 2] + q[1, 3], 0.0),
            (q[0, 1] + q[2, 3], 1.0 - q[0, 0] - q[2, 2], q[1, 2] - q[0, 3], 0.0),
            (q[0, 2] - q[1, 3], q[1, 2] + q[0, 3], 1.0 - q[0, 0] - q[1, 1], 0.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )


def translation_matrix(direction: np.ndarray) -> np.ndarray:
    M = np.identity(4)
    M[:3, 3] = direction[:3]
    return M


def quaternion_about_axis(angle: float, axis: tuple[float, float, float]) -> np.ndarray:
    quaternion = np.zeros((4,), dtype=np.float64)
    quaternion[:3] = axis[:3]
    qlen = np.linalg.norm(quaternion)
    if qlen > _EPS:
        quaternion *= math.sin(angle / 2.0) / qlen
    quaternion[3] = math.cos(angle / 2.0)
    return quaternion
