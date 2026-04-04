"""Apply Savitzky-Golay filter (polynomial order 2, window 7) to a signal.

Signal is v[i] = sin(i*0.01) + 0.1*((i*7+3)%100-50)/50.0. Returns sum of smoothed values.

Keywords: signal processing, Savitzky-Golay, smoothing, convolution, filter
"""

import math

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(500000,))
def savitzky_golay(n: int) -> float:
    """Apply Savitzky-Golay filter to a noisy signal and return sum of smoothed values.

    Args:
        n: Signal length.

    Returns:
        Sum of smoothed values.
    """
    # Precompute convolution coefficients for order 2, window 7
    # These are the standard SG coefficients: [-2, 3, 6, 7, 6, 3, -2] / 21
    coeffs = [-2.0 / 21.0, 3.0 / 21.0, 6.0 / 21.0, 7.0 / 21.0, 6.0 / 21.0, 3.0 / 21.0, -2.0 / 21.0]
    half_w = 3  # half window size

    # Generate signal
    signal = [0.0] * n
    for i in range(n):
        signal[i] = math.sin(i * 0.01) + 0.1 * ((i * 7 + 3) % 100 - 50) / 50.0

    # Apply filter (skip edges where window doesn't fit)
    total = 0.0
    for i in range(half_w, n - half_w):
        val = 0.0
        for j in range(7):
            val += coeffs[j] * signal[i - half_w + j]
        total += val

    # Add unfiltered edge values
    for i in range(half_w):
        total += signal[i]
    for i in range(n - half_w, n):
        total += signal[i]

    return total
