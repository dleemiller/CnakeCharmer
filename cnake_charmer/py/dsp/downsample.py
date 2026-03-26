"""Downsample a signal by factor 4 with anti-alias FIR filter.

Applies a 21-tap low-pass FIR filter before decimation and returns the
sum of the downsampled signal.

Keywords: dsp, downsample, decimate, anti-alias, FIR, filter, benchmark
"""

import math

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(1000000,))
def downsample(n: int) -> float:
    """Downsample signal by factor 4 with anti-alias filter, return sum.

    Signal: s[i] = sin(i*0.01) + sin(i*0.005).
    Anti-alias filter: 21-tap sinc * Hamming, cutoff pi/4.

    Args:
        n: Input signal length.

    Returns:
        Sum of the downsampled signal.
    """
    taps = 21
    mid = taps // 2
    factor = 4

    # Design anti-alias filter coefficients
    h = [0.0] * taps
    cutoff = 1.0 / factor  # Normalized cutoff
    for k in range(taps):
        if k == mid:
            h[k] = cutoff
        else:
            diff = k - mid
            h[k] = math.sin(math.pi * cutoff * diff) / (math.pi * diff)
        # Hamming window
        h[k] *= 0.54 - 0.46 * math.cos(2.0 * math.pi * k / (taps - 1))

    # Generate signal
    s = [math.sin(i * 0.01) + math.sin(i * 0.005) for i in range(n)]

    # Filter and downsample
    total = 0.0
    for i in range(mid, n - mid, factor):
        acc = 0.0
        for k in range(taps):
            acc += h[k] * s[i - mid + k]
        total += acc

    return total
