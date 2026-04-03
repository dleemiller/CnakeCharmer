"""2D Gaussian integration over an n x n grid.

Evaluate the integral of a 2D Gaussian function over a square domain
using direct grid summation (Riemann sum).

Keywords: numerical, gaussian, integration, 2d, grid, summation, benchmark
"""

import math

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(800,))
def gaussian_integral_2d(n: int) -> tuple:
    """Integrate a 2D Gaussian over [-3, 3] x [-3, 3] on an n x n grid.

    The Gaussian has FWHM=2.0 (sigma = FWHM / 2.3548).
    Uses a Riemann sum with cell areas dx * dy.

    Args:
        n: Number of grid points along each axis.

    Returns:
        Tuple of (integral_value, max_contribution) where max_contribution
        is the largest single cell contribution.
    """
    fwhm = 2.0
    sigma = fwhm / 2.3548
    inv_2sigma2 = 1.0 / (2.0 * sigma * sigma)

    low = -3.0
    high = 3.0
    dx = (high - low) / n
    dy = (high - low) / n
    da = dx * dy

    integral = 0.0
    max_contrib = 0.0

    for i in range(n):
        y = low + (i + 0.5) * dy
        y2 = y * y
        for j in range(n):
            x = low + (j + 0.5) * dx
            val = math.exp(-(x * x + y2) * inv_2sigma2)
            contrib = val * da
            integral += contrib
            if contrib > max_contrib:
                max_contrib = contrib

    return (integral, max_contrib)
