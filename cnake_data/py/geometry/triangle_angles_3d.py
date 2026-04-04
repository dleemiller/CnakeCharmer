"""Compute angles of triangles in 3D space using law of cosines.

For n deterministic triangles with vertices in 3D, compute all three
interior angles at each vertex using the law of cosines.

Keywords: geometry, triangle, angles, 3d, law of cosines, dot product, benchmark
"""

import math

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(200000,))
def triangle_angles_3d(n: int) -> tuple:
    """Compute angles for n triangles in 3D and return statistics.

    Each triangle has vertices generated deterministically:
        A = (k*0.3, k*0.7+1, k*0.2)
        B = (k*0.5+2, k*0.1, k*0.9+1)
        C = (k*0.1+1, k*0.4+2, k*0.6)
    where k = i % 1000 for triangle i.

    Args:
        n: Number of triangles to process.

    Returns:
        Tuple of (sum_angles, min_angle) in degrees.
    """
    sum_angles = 0.0
    min_angle = 360.0

    for i in range(n):
        k = (i % 1000) * 0.001

        # Vertices
        ax = k * 300.0
        ay = k * 700.0 + 1.0
        az = k * 200.0

        bx = k * 500.0 + 2.0
        by = k * 100.0
        bz = k * 900.0 + 1.0

        cx = k * 100.0 + 1.0
        cy = k * 400.0 + 2.0
        cz = k * 600.0

        # Side lengths squared
        ab2 = (bx - ax) ** 2 + (by - ay) ** 2 + (bz - az) ** 2
        bc2 = (cx - bx) ** 2 + (cy - by) ** 2 + (cz - bz) ** 2
        ca2 = (ax - cx) ** 2 + (ay - cy) ** 2 + (az - cz) ** 2

        ab = math.sqrt(ab2)
        bc = math.sqrt(bc2)
        ca = math.sqrt(ca2)

        # Avoid degenerate triangles
        if ab < 1e-12 or bc < 1e-12 or ca < 1e-12:
            continue

        # Angles via law of cosines: cos(A) = (ab^2 + ca^2 - bc^2) / (2*ab*ca)
        cos_a = (ab2 + ca2 - bc2) / (2.0 * ab * ca)
        cos_b = (ab2 + bc2 - ca2) / (2.0 * ab * bc)
        cos_c = (bc2 + ca2 - ab2) / (2.0 * bc * ca)

        # Clamp to [-1, 1] for numerical safety
        cos_a = max(-1.0, min(1.0, cos_a))
        cos_b = max(-1.0, min(1.0, cos_b))
        cos_c = max(-1.0, min(1.0, cos_c))

        angle_a = math.acos(cos_a) * 180.0 / math.pi
        angle_b = math.acos(cos_b) * 180.0 / math.pi
        angle_c = math.acos(cos_c) * 180.0 / math.pi

        sum_angles += angle_a + angle_b + angle_c

        for angle in (angle_a, angle_b, angle_c):
            if angle < min_angle:
                min_angle = angle

    return (sum_angles, min_angle)
