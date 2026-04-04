"""Evaluate a polynomial at n Chebyshev nodes and return the sum.

Keywords: chebyshev, polynomial, numerical, interpolation, benchmark
"""

import math

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(1000000,))
def chebyshev_nodes(n: int) -> float:
    """Evaluate polynomial p(x) at n Chebyshev nodes and return sum.

    Chebyshev nodes: x_k = cos((2k+1)/(2n) * pi) for k=0..n-1.
    Polynomial: p(x) = sum(sin(k*0.1) * x^k for k=0..9).

    Args:
        n: Number of Chebyshev nodes.

    Returns:
        Sum of p(x_k) for all Chebyshev nodes.
    """
    # Precompute polynomial coefficients
    coeffs = [math.sin(k * 0.1) for k in range(10)]

    total = 0.0
    for k in range(n):
        x = math.cos((2 * k + 1) / (2 * n) * math.pi)

        # Evaluate polynomial using Horner's method
        val = coeffs[9]
        for j in range(8, -1, -1):
            val = val * x + coeffs[j]

        total += val

    return total
