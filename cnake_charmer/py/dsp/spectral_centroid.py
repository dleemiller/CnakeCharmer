"""Compute spectral centroid of a signal using DFT magnitudes.

The spectral centroid is the weighted mean of frequencies, weighted by
their magnitudes. Returns the centroid as a frequency bin index.

Keywords: dsp, spectral, centroid, DFT, frequency, magnitude, benchmark
"""

import math

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(10000,))
def spectral_centroid(n: int) -> float:
    """Compute spectral centroid and return as frequency bin float.

    Signal: s[i] = sin(i*0.1) + 0.3*sin(i*0.3).

    Args:
        n: Signal length.

    Returns:
        Spectral centroid (frequency bin index as float).
    """
    two_pi_over_n = 2.0 * math.pi / n

    # Compute DFT magnitudes for bins 0..n//2
    half_n = n // 2
    weighted_sum = 0.0
    magnitude_sum = 0.0

    for k in range(half_n + 1):
        re = 0.0
        im = 0.0
        for i in range(n):
            sig = math.sin(i * 0.1) + 0.3 * math.sin(i * 0.3)
            angle = two_pi_over_n * k * i
            re += sig * math.cos(angle)
            im -= sig * math.sin(angle)
        mag = math.sqrt(re * re + im * im)
        weighted_sum += k * mag
        magnitude_sum += mag

    if magnitude_sum == 0.0:
        return 0.0
    return weighted_sum / magnitude_sum
