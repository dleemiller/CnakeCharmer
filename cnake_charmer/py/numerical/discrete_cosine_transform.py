"""Discrete Cosine Transform (DCT-II).

Keywords: numerical, DCT, cosine, transform, signal processing, benchmark
"""

import math

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(2000,))
def discrete_cosine_transform(n: int) -> tuple:
    """Compute the DCT-II of a deterministic signal.

    Input signal: x[i] = sin(i * 0.05) + 0.5 * cos(i * 0.13).
    Computes DCT-II coefficients using the direct O(n^2) formula and returns
    summary statistics.

    Args:
        n: Signal length.

    Returns:
        Tuple of (sum of coefficients, max coefficient, coefficient at index n//4).
    """
    # Build input signal
    x = [0.0] * n
    for i in range(n):
        x[i] = math.sin(i * 0.05) + 0.5 * math.cos(i * 0.13)

    # Compute DCT-II: X[k] = sum_{i=0}^{n-1} x[i] * cos(pi/n * (i + 0.5) * k)
    pi_over_n = math.pi / n
    coeff_sum = 0.0
    coeff_max = -1e300
    coeff_quarter = 0.0

    for k in range(n):
        val = 0.0
        for i in range(n):
            val += x[i] * math.cos(pi_over_n * (i + 0.5) * k)
        coeff_sum += val
        if val > coeff_max:
            coeff_max = val
        if k == n // 4:
            coeff_quarter = val

    return (coeff_sum, coeff_max, coeff_quarter)
