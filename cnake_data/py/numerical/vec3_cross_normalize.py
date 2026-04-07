"""Batch 3D vector cross-product and normalization.

For n pairs of deterministically generated unit-ish vectors, computes
the cross product and normalizes the result, accumulating a checksum.

Keywords: numerical, vector, cross product, normalize, 3d, linear algebra, geometry
"""

import math

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(8000,))
def vec3_cross_normalize(n: int) -> float:
    """Cross-product + normalize n deterministic vector pairs, return checksum.

    Args:
        n: Number of vector pairs to process.

    Returns:
        Sum of all normalized cross-product components (a float checksum).
    """
    total = 0.0
    for i in range(n):
        fi = float(i)
        # Two deterministic vectors
        ax = math.sin(fi * 0.7)
        ay = math.cos(fi * 0.3)
        az = math.sin(fi * 1.1)
        bx = math.cos(fi * 0.5)
        by = math.sin(fi * 0.9)
        bz = math.cos(fi * 1.3)

        # Cross product
        cx = ay * bz - az * by
        cy = az * bx - ax * bz
        cz = ax * by - ay * bx

        # Normalize
        norm = math.sqrt(cx * cx + cy * cy + cz * cz)
        if norm > 0.0:
            cx /= norm
            cy /= norm
            cz /= norm

        total += cx + cy + cz
    return total
