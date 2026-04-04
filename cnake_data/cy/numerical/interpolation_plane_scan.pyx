# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Scan planar interpolation residual statistics over a rectangular grid (Cython)."""

from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(0.17, -0.05, 1.3, -64, 64, -48, 48, 4, 0.73))
def interpolation_plane_scan(
    double a,
    double b,
    double c,
    int x_start,
    int x_stop,
    int y_start,
    int y_stop,
    int passes,
    double blend,
):
    cdef int p, x, y
    cdef double bias, z, observed, err, abs_err
    cdef double sum_interp = 0.0
    cdef double weighted_error = 0.0
    cdef double max_abs_error = 0.0

    for p in range(passes):
        bias = (p + 1) * 0.03125
        for y in range(y_start, y_stop):
            for x in range(x_start, x_stop):
                z = a * x + b * y + c
                observed = z * blend + (x - y) * 0.25 + bias
                err = observed - z
                if err >= 0.0:
                    abs_err = err
                else:
                    abs_err = -err
                sum_interp += z
                weighted_error += abs_err * (1.0 + ((x + y + p) & 3) * 0.125)
                if abs_err > max_abs_error:
                    max_abs_error = abs_err

    return (sum_interp, weighted_error, max_abs_error)
