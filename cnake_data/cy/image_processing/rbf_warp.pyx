# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Gaussian radial basis function (RBF) image warp — Cython implementation."""

from libc.math cimport cos, exp, sin, sqrt
from libc.stdlib cimport free, malloc

from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(60, 60, 12, 20.0))
def rbf_warp(int rows, int cols, int n_ctrl, double sigma):
    """Evaluate Gaussian RBF warp displacement at a grid of pixels."""
    cdef int n_side = <int>sqrt(<double>n_ctrl)
    cdef int m = n_side * n_side

    cdef double *ctrl_x = <double *>malloc(m * sizeof(double))
    cdef double *ctrl_y = <double *>malloc(m * sizeof(double))
    cdef double *wx = <double *>malloc(m * sizeof(double))
    cdef double *wy = <double *>malloc(m * sizeof(double))
    if not ctrl_x or not ctrl_y or not wx or not wy:
        free(ctrl_x); free(ctrl_y); free(wx); free(wy)
        raise MemoryError()

    cdef double cell_w = (cols - 1.0) / (n_side - 1) if n_side > 1 else cols / 2.0
    cdef double cell_h = (rows - 1.0) / (n_side - 1) if n_side > 1 else rows / 2.0
    cdef double inv_s2 = 1.0 / (sigma * sigma)
    cdef int i, j, k

    k = 0
    for i in range(n_side):
        for j in range(n_side):
            ctrl_x[k] = j * cell_w
            ctrl_y[k] = i * cell_h
            wx[k] = 3.0 * sin(i * 0.8 + j * 0.5)
            wy[k] = 2.0 * cos(i * 0.6 - j * 0.7)
            k += 1

    cdef double sum_dx = 0.0, sum_dy = 0.0, max_disp = 0.0
    cdef double x, y, dx, dy, ex, ey, r2, kernel, disp
    cdef int r, c

    for r in range(rows):
        y = <double>r
        for c in range(cols):
            x = <double>c
            dx = 0.0
            dy = 0.0
            for k in range(m):
                ex = x - ctrl_x[k]
                ey = y - ctrl_y[k]
                r2 = ex * ex + ey * ey
                kernel = exp(-r2 * inv_s2)
                dx += wx[k] * kernel
                dy += wy[k] * kernel
            sum_dx += dx
            sum_dy += dy
            disp = sqrt(dx * dx + dy * dy)
            if disp > max_disp:
                max_disp = disp

    free(ctrl_x); free(ctrl_y); free(wx); free(wy)
    return (sum_dx, sum_dy, max_disp)
