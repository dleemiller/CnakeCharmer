"""Vectorized ray-triangle intersection helpers (Moller-Trumbore)."""

from __future__ import annotations

import math

import numpy as np


def fast_cross_mm(a: np.ndarray, b: np.ndarray, r: np.ndarray) -> None:
    n = r.shape[0]
    for i in range(n):
        r[i, 0] = a[i, 1] * b[i, 2] - a[i, 2] * b[i, 1]
        r[i, 1] = a[i, 2] * b[i, 0] - a[i, 0] * b[i, 2]
        r[i, 2] = a[i, 0] * b[i, 1] - a[i, 1] * b[i, 0]


def fast_cross_vm(a: np.ndarray, b: np.ndarray, r: np.ndarray) -> None:
    n = r.shape[0]
    for i in range(n):
        r[i, 0] = a[1] * b[i, 2] - a[2] * b[i, 1]
        r[i, 1] = a[2] * b[i, 0] - a[0] * b[i, 2]
        r[i, 2] = a[0] * b[i, 1] - a[1] * b[i, 0]


def fast_cross_vv(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.array(
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ],
        dtype=np.float32,
    )


def fast_ray_triangles_intersection(
    origin: np.ndarray, destination: np.ndarray, p0: np.ndarray, p1: np.ndarray, p2: np.ndarray
) -> np.ndarray:
    ray = destination - origin
    norm = math.sqrt(float(np.dot(ray, ray)))
    ray = ray / norm

    N = p0.shape[0]
    eps = 1e-6
    results = np.empty((100, 3), dtype=np.float32)
    rs_cnt = 0

    for i in range(N):
        e1 = p1[i] - p0[i]
        e2 = p2[i] - p0[i]
        pvec = fast_cross_vv(ray, e2)
        det = float(np.dot(e1, pvec))
        if -eps < det < eps:
            continue
        inv_det = 1.0 / det
        tvec = origin - p0[i]
        u = float(np.dot(tvec, pvec)) * inv_det
        if u < 0 or u > 1:
            continue
        qvec = fast_cross_vv(tvec, e1)
        v = float(np.dot(ray, qvec)) * inv_det
        if v < 0 or u + v > 1:
            continue
        t = float(np.dot(e2, qvec)) * inv_det
        results[rs_cnt] = origin + t * ray
        rs_cnt += 1
        if rs_cnt >= 100:
            break

    return results[:rs_cnt]
