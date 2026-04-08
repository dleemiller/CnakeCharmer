# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Sequential paraxial ray tracing through a multi-element lens system — Cython implementation."""

from libc.math cimport fabs

from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(5000, 8))
def ray_trace_sequential(int n_rays, int n_surfaces):
    """Trace n_rays through an n_surfaces lens system using ABCD matrices."""
    cdef double d = 10.0
    cdef double f = 50.0
    cdef double sum_y = 0.0, sum_u = 0.0
    cdef int n_focused = 0
    cdef double u0, y, u
    cdef int i, s

    for i in range(n_rays):
        u0 = -0.1 + 0.2 * i / (n_rays - 1) if n_rays > 1 else 0.0
        y = 1.0
        u = u0
        for s in range(n_surfaces):
            if s % 2 == 0:
                y = y + d * u
            else:
                u = u - y / f
        sum_y += y
        sum_u += u
        if fabs(y) < 5.0:
            n_focused += 1

    return (sum_y, sum_u, n_focused)
