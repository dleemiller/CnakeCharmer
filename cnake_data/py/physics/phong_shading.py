"""Phong illumination model for a surface normal field.

Computes full Phong shading (ambient + diffuse + specular) for every pixel
in a spherical normal map, given a fixed light and view direction.

Keywords: Phong shading, specular reflection, BRDF, illumination model, rendering
"""

import math

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(150, 150))
def phong_shading(rows: int, cols: int) -> tuple:
    """Apply Phong illumination to a deterministic spherical normal field.

    Light direction L = (1,1,1)/√3 (diagonal).  View direction V = (0,0,1) (camera).
    Specular exponent n=16.  Material: ka=0.1, kd=0.6, ks=0.3.

    Args:
        rows: Image height.
        cols: Image width.

    Returns:
        Tuple of (total_intensity, specular_sum, diffuse_sum).
    """
    lx = 0.5773502691896258
    ly = 0.5773502691896258
    lz = 0.5773502691896258
    ka = 0.1
    kd = 0.6
    ks = 0.3

    total = 0.0
    specular_sum = 0.0
    diffuse_sum = 0.0

    for r in range(rows):
        theta = math.pi * r / (rows - 1) if rows > 1 else 0.0
        sin_t = math.sin(theta)
        cos_t = math.cos(theta)
        for c in range(cols):
            phi = 2.0 * math.pi * c / (cols - 1) if cols > 1 else 0.0
            nx = sin_t * math.cos(phi)
            ny = sin_t * math.sin(phi)
            nz = cos_t

            ndotl = nx * lx + ny * ly + nz * lz
            diff = kd * ndotl if ndotl > 0.0 else 0.0

            if ndotl > 0.0:
                # Reflection R = 2*(N·L)*N - L; V=(0,0,1) so R·V = Rz
                rdotv = 2.0 * ndotl * nz - lz
                if rdotv > 0.0:
                    # Specular: (R·V)^16 via repeated squaring
                    rdotv_sq = rdotv * rdotv
                    rdotv_4 = rdotv_sq * rdotv_sq
                    rdotv_8 = rdotv_4 * rdotv_4
                    rdotv_16 = rdotv_8 * rdotv_8
                    spec = ks * rdotv_16
                else:
                    spec = 0.0
            else:
                spec = 0.0

            irr = ka + diff + spec
            total += irr
            diffuse_sum += diff
            specular_sum += spec

    return (total, specular_sum, diffuse_sum)
