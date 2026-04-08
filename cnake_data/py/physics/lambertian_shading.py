"""Lambertian diffuse shading for a surface normal field.

Computes Lambertian (diffuse) reflectance for every pixel in an n×m normal
map, given a fixed light direction. Models ideal matte surface reflectance.

Keywords: Lambertian BRDF, diffuse shading, normal map, photometric, rendering
"""

import math

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(150, 150))
def lambertian_shading(rows: int, cols: int) -> tuple:
    """Compute Lambertian shading for a deterministic spherical normal field.

    The normal at pixel (r, c) is derived from a sphere-like parameterization.
    Light direction: (0.577, 0.577, 0.577) (normalized 45° diagonal).

    Args:
        rows: Image height.
        cols: Image width.

    Returns:
        Tuple of (total_irradiance, lit_count, max_irradiance).
    """
    # Normalized light direction
    lx = 0.5773502691896258
    ly = 0.5773502691896258
    lz = 0.5773502691896258

    total = 0.0
    lit_count = 0
    max_irr = 0.0

    for r in range(rows):
        # Spherical normal parameterization
        theta = math.pi * r / (rows - 1) if rows > 1 else 0.0
        sin_t = math.sin(theta)
        cos_t = math.cos(theta)
        for c in range(cols):
            phi = 2.0 * math.pi * c / (cols - 1) if cols > 1 else 0.0
            nx = sin_t * math.cos(phi)
            ny = sin_t * math.sin(phi)
            nz = cos_t
            # Lambertian: irradiance = max(0, N · L)
            dot = nx * lx + ny * ly + nz * lz
            irr = dot if dot > 0.0 else 0.0
            total += irr
            if irr > 0.0:
                lit_count += 1
            if irr > max_irr:
                max_irr = irr

    return (total, lit_count, max_irr)
