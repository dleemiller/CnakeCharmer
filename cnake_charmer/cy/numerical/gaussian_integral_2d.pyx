# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""2D Gaussian integration over an n x n grid (Cython-optimized).

Keywords: numerical, gaussian, integration, 2d, grid, summation, cython, benchmark
"""

from libc.math cimport exp
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(800,))
def gaussian_integral_2d(int n):
    """Integrate a 2D Gaussian over [-3, 3] x [-3, 3] on an n x n grid.

    Args:
        n: Number of grid points along each axis.

    Returns:
        Tuple of (integral_value, max_contribution).
    """
    cdef double fwhm = 2.0
    cdef double sigma = fwhm / 2.3548
    cdef double inv_2sigma2 = 1.0 / (2.0 * sigma * sigma)

    cdef double low = -3.0
    cdef double high = 3.0
    cdef double dx = (high - low) / n
    cdef double dy = (high - low) / n
    cdef double da = dx * dy

    cdef double integral = 0.0
    cdef double max_contrib = 0.0
    cdef double x, y, y2, val, contrib
    cdef int i, j

    for i in range(n):
        y = low + (i + 0.5) * dy
        y2 = y * y
        for j in range(n):
            x = low + (j + 0.5) * dx
            val = exp(-(x * x + y2) * inv_2sigma2)
            contrib = val * da
            integral += contrib
            if contrib > max_contrib:
                max_contrib = contrib

    return (integral, max_contrib)
