# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Camera response function calibration — Cython implementation."""

from libc.math cimport cos, sin

from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(150, 150))
def camera_response(int rows, int cols):
    """Apply and invert a polynomial camera response function."""
    cdef double sum_linear = 0.0, sum_nonlinear = 0.0
    cdef int n_clipped = 0
    cdef double irr, irr2, irr3, cam, x, fx, dfx
    cdef int r, c, it

    for r in range(rows):
        for c in range(cols):
            irr = (sin(r * 0.05) * 0.5 + 0.5) * (cos(c * 0.07) * 0.5 + 0.5)

            irr2 = irr * irr
            irr3 = irr2 * irr
            cam = irr - 0.3 * irr2 + 0.1 * irr3

            x = cam
            for it in range(5):
                fx = x - 0.3 * x * x + 0.1 * x * x * x - cam
                dfx = 1.0 - 0.6 * x + 0.3 * x * x
                x = x - fx / dfx

            sum_linear += x
            sum_nonlinear += cam
            if cam > 0.9:
                n_clipped += 1

    return (sum_linear, sum_nonlinear, n_clipped)
