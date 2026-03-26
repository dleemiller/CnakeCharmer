"""Compute pi using the Gauss-Legendre algorithm.

Keywords: numerical, pi, gauss-legendre, iterative, convergence, benchmark
"""

import math

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(50,))
def gauss_legendre_pi(n: int) -> float:
    """Compute pi using n iterations of the Gauss-Legendre algorithm.

    Args:
        n: Number of iterations.

    Returns:
        Approximation of pi as a float.
    """
    a = 1.0
    b = 1.0 / math.sqrt(2.0)
    t = 0.25
    p = 1.0

    for _ in range(n):
        a_next = (a + b) / 2.0
        b = math.sqrt(a * b)
        t = t - p * (a - a_next) * (a - a_next)
        p = 2.0 * p
        a = a_next

    result = (a + b) * (a + b) / (4.0 * t)
    return result
