"""Naive DFT (Discrete Fourier Transform) of n complex values.

Computes the DFT using the O(n^2) direct summation formula and returns the
sum of magnitudes of all frequency bins.

Keywords: numerical, DFT, Fourier, transform, frequency, magnitude, benchmark
"""

import math

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(2000,))
def fft_naive(n: int) -> float:
    """Compute naive DFT of n complex values and return sum of magnitudes.

    Input signal: real[i] = sin(i * 0.1), imag[i] = 0.

    Args:
        n: Number of samples.

    Returns:
        Sum of magnitudes of all frequency bins.
    """
    real_in = [math.sin(i * 0.1) for i in range(n)]

    total = 0.0
    two_pi_over_n = 2.0 * math.pi / n

    for k in range(n):
        re = 0.0
        im = 0.0
        for j in range(n):
            angle = two_pi_over_n * k * j
            re += real_in[j] * math.cos(angle)
            im -= real_in[j] * math.sin(angle)
        total += math.sqrt(re * re + im * im)

    return total
