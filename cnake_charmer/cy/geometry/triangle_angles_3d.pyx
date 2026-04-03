# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Compute angles of triangles in 3D space using law of cosines (Cython-optimized).

Keywords: geometry, triangle, angles, 3d, law of cosines, dot product, cython, benchmark
"""

from libc.math cimport sqrt, acos
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(200000,))
def triangle_angles_3d(int n):
    """Compute angles for n triangles in 3D and return statistics.

    Args:
        n: Number of triangles to process.

    Returns:
        Tuple of (sum_angles, min_angle) in degrees.
    """
    cdef double sum_angles = 0.0
    cdef double min_angle = 360.0
    cdef double k
    cdef double ax, ay, az, bx, by, bz, cx, cy, cz
    cdef double ab2, bc2, ca2, ab, bc, ca
    cdef double cos_a, cos_b, cos_c
    cdef double angle_a, angle_b, angle_c
    cdef double pi_over_180 = 180.0 / 3.14159265358979323846
    cdef int i

    for i in range(n):
        k = (i % 1000) * 0.001

        ax = k * 300.0
        ay = k * 700.0 + 1.0
        az = k * 200.0

        bx = k * 500.0 + 2.0
        by = k * 100.0
        bz = k * 900.0 + 1.0

        cx = k * 100.0 + 1.0
        cy = k * 400.0 + 2.0
        cz = k * 600.0

        ab2 = (bx - ax) * (bx - ax) + (by - ay) * (by - ay) + (bz - az) * (bz - az)
        bc2 = (cx - bx) * (cx - bx) + (cy - by) * (cy - by) + (cz - bz) * (cz - bz)
        ca2 = (ax - cx) * (ax - cx) + (ay - cy) * (ay - cy) + (az - cz) * (az - cz)

        ab = sqrt(ab2)
        bc = sqrt(bc2)
        ca = sqrt(ca2)

        if ab < 1e-12 or bc < 1e-12 or ca < 1e-12:
            continue

        cos_a = (ab2 + ca2 - bc2) / (2.0 * ab * ca)
        cos_b = (ab2 + bc2 - ca2) / (2.0 * ab * bc)
        cos_c = (bc2 + ca2 - ab2) / (2.0 * bc * ca)

        # Clamp
        if cos_a < -1.0:
            cos_a = -1.0
        elif cos_a > 1.0:
            cos_a = 1.0
        if cos_b < -1.0:
            cos_b = -1.0
        elif cos_b > 1.0:
            cos_b = 1.0
        if cos_c < -1.0:
            cos_c = -1.0
        elif cos_c > 1.0:
            cos_c = 1.0

        angle_a = acos(cos_a) * pi_over_180
        angle_b = acos(cos_b) * pi_over_180
        angle_c = acos(cos_c) * pi_over_180

        sum_angles += angle_a + angle_b + angle_c

        if angle_a < min_angle:
            min_angle = angle_a
        if angle_b < min_angle:
            min_angle = angle_b
        if angle_c < min_angle:
            min_angle = angle_c

    return (sum_angles, min_angle)
