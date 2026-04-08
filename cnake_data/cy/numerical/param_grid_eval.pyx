# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Parametric function evaluation on a 4D grid — Cython implementation."""

from libc.math cimport exp, sin
from libc.stdlib cimport free, malloc

from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(12, 10))
def param_grid_eval(int n_h, int n_xyz):
    """Evaluate parametric function over h-grid with (x,y,z) inner loop.

    Args:
        n_h: Number of h (parameter) values in [0.5, 2.5].
        n_xyz: Number of x/y/z points per axis in [-1, 1].

    Returns:
        Tuple of (total_sum, first_val, last_val) from the output array.
    """
    cdef double *out = <double *>malloc(n_h * sizeof(double))
    if not out:
        raise MemoryError()

    cdef double h_min = 0.5
    cdef double h_max = 2.5
    cdef double x_min = -1.0
    cdef double x_max = 1.0
    cdef double h, X, Y, Z, result, arg, inv_nh1, inv_nxyz1
    cdef int h_i, x_i, y_i, z_i

    inv_nh1 = (h_max - h_min) / (n_h - 1) if n_h > 1 else 0.0
    inv_nxyz1 = (x_max - x_min) / (n_xyz - 1) if n_xyz > 1 else 0.0

    for h_i in range(n_h):
        h = h_min + h_i * inv_nh1
        result = 0.0
        for x_i in range(n_xyz):
            X = x_min + x_i * inv_nxyz1
            for y_i in range(n_xyz):
                Y = x_min + y_i * inv_nxyz1
                for z_i in range(n_xyz):
                    Z = x_min + z_i * inv_nxyz1
                    arg = 2.0 * Z + X + Y * Y - h
                    result += exp(-(arg * arg)) * (sin(X + Y + 3.0 * Z + h) + (Y + Z + h) * (Y + Z + h))
        out[h_i] = result

    cdef double total = 0.0
    for h_i in range(n_h):
        total += out[h_i]
    cdef double first_val = out[0]
    cdef double last_val = out[n_h - 1]
    free(out)
    return (total, first_val, last_val)
