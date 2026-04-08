# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Phong illumination model for a surface normal field — Cython implementation."""

from libc.math cimport cos, pi, sin

from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(150, 150))
def phong_shading(int rows, int cols):
    """Apply Phong illumination to a deterministic spherical normal field."""
    cdef double lx = 0.5773502691896258
    cdef double ly = 0.5773502691896258
    cdef double lz = 0.5773502691896258
    cdef double ka = 0.1, kd = 0.6, ks = 0.3

    cdef double total = 0.0, specular_sum = 0.0, diffuse_sum = 0.0
    cdef double theta, sin_t, cos_t, phi, nx, ny, nz
    cdef double ndotl, diff, rdotv, spec, irr
    cdef double rdotv_sq, rdotv_4, rdotv_8, rdotv_16
    cdef int r, c

    for r in range(rows):
        theta = pi * r / (rows - 1) if rows > 1 else 0.0
        sin_t = sin(theta)
        cos_t = cos(theta)
        for c in range(cols):
            phi = 2.0 * pi * c / (cols - 1) if cols > 1 else 0.0
            nx = sin_t * cos(phi)
            ny = sin_t * sin(phi)
            nz = cos_t

            ndotl = nx * lx + ny * ly + nz * lz
            diff = kd * ndotl if ndotl > 0.0 else 0.0

            if ndotl > 0.0:
                # R·V with V=(0,0,1): R_z = 2*(N·L)*N_z - L_z
                rdotv = 2.0 * ndotl * nz - lz
                if rdotv > 0.0:
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
