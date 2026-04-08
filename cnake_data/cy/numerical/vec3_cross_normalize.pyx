# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Batch 3D vector cross-product and normalization — Cython implementation."""

from libc.math cimport cos, sin, sqrt

from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(8000,))
def vec3_cross_normalize(int n):
    """Cross-product + normalize n deterministic vector pairs, return checksum.

    Args:
        n: Number of vector pairs to process.

    Returns:
        Sum of all normalized cross-product components (a float checksum).
    """
    cdef int i
    cdef double fi, ax, ay, az, bx, by, bz
    cdef double cx, cy, cz, norm, total
    cdef double first_norm = 0.0, last_cx = 0.0

    total = 0.0
    for i in range(n):
        fi = <double>i
        ax = sin(fi * 0.7)
        ay = cos(fi * 0.3)
        az = sin(fi * 1.1)
        bx = cos(fi * 0.5)
        by = sin(fi * 0.9)
        bz = cos(fi * 1.3)

        cx = ay * bz - az * by
        cy = az * bx - ax * bz
        cz = ax * by - ay * bx

        norm = sqrt(cx * cx + cy * cy + cz * cz)
        if norm > 0.0:
            cx /= norm
            cy /= norm
            cz /= norm

        total += cx + cy + cz
        if i == 0:
            first_norm = norm
        last_cx = cx
    return (total, first_norm, last_cx)
