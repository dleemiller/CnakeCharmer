"""Apply Hanning, Hamming, and Blackman windows to a signal.

Computes all three windowed signals and returns the sum of all values.

Keywords: dsp, window, Hanning, Hamming, Blackman, signal, benchmark
"""

import math

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(1000000,))
def window_functions(n: int) -> float:
    """Apply three window functions and return sum of all windowed values.

    Signal: s[i] = sin(i*0.1).

    Args:
        n: Signal length.

    Returns:
        Sum of Hanning + Hamming + Blackman windowed signals.
    """
    two_pi_over_nm1 = 2.0 * math.pi / (n - 1)
    four_pi_over_nm1 = 4.0 * math.pi / (n - 1)
    total = 0.0

    for i in range(n):
        sig = math.sin(i * 0.1)
        cos_val = math.cos(two_pi_over_nm1 * i)
        cos_val2 = math.cos(four_pi_over_nm1 * i)

        # Hanning
        hanning = 0.5 * (1.0 - cos_val)
        # Hamming
        hamming = 0.54 - 0.46 * cos_val
        # Blackman
        blackman = 0.42 - 0.5 * cos_val + 0.08 * cos_val2

        total += sig * hanning + sig * hamming + sig * blackman

    return total
