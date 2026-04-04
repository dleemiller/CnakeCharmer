"""Count zero crossings in a noisy signal.

Returns the total number of sign changes in the signal.

Keywords: dsp, zero-crossing, rate, sign, signal, benchmark
"""

import math

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(3000000,))
def zero_crossing_rate(n: int) -> int:
    """Count zero crossings in a signal.

    Signal: s[i] = sin(i*0.1)*cos(i*0.03) + 0.1*((i*7+3)%100-50)/50.0

    Args:
        n: Signal length.

    Returns:
        Number of zero crossings.
    """
    count = 0
    prev = math.sin(0.0) * math.cos(0.0) + 0.1 * ((0 * 7 + 3) % 100 - 50) / 50.0

    for i in range(1, n):
        curr = math.sin(i * 0.1) * math.cos(i * 0.03) + 0.1 * ((i * 7 + 3) % 100 - 50) / 50.0
        if (prev >= 0.0 and curr < 0.0) or (prev < 0.0 and curr >= 0.0):
            count += 1
        prev = curr

    return count
