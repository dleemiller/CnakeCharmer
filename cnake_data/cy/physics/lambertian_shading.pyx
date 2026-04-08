# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Lambertian diffuse shading for a surface normal field — Cython implementation."""

from libc.math cimport cos, pi, sin

from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(150, 150))
def lambertian_shading(int rows, int cols):
    """Compute Lambertian shading for a deterministic spherical normal field."""
    cdef double lx = 0.5773502691896258
    cdef double ly = 0.5773502691896258
    cdef double lz = 0.5773502691896258

    cdef double total = 0.0
    cdef int lit_count = 0
    cdef double max_irr = 0.0
    cdef double theta, sin_t, cos_t, phi, nx, ny, nz, dot, irr
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
            dot = nx * lx + ny * ly + nz * lz
            irr = dot if dot > 0.0 else 0.0
            total += irr
            if irr > 0.0:
                lit_count += 1
            if irr > max_irr:
                max_irr = irr

    return (total, lit_count, max_irr)
