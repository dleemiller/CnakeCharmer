"""Apply a 31-tap low-pass FIR filter to a sinusoidal signal.

Coefficients use a sinc-Hamming window design. Returns the sum of the
filtered output signal.

Keywords: dsp, FIR, filter, convolution, low-pass, Hamming, signal, benchmark
"""

import math

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(500000,))
def fir_filter(n: int) -> float:
    """Apply a 31-tap FIR filter and return sum of filtered signal.

    Signal: s[i] = sin(i*0.01) + 0.5*sin(i*0.1).
    Filter: sinc * Hamming window, cutoff 0.2*pi.

    Args:
        n: Signal length.

    Returns:
        Sum of the filtered signal.
    """
    taps = 31
    mid = taps // 2

    # Design FIR coefficients: sinc * Hamming
    h = [0.0] * taps
    for k in range(taps):
        if k == mid:
            h[k] = 0.2
        else:
            diff = k - mid
            h[k] = math.sin(0.2 * math.pi * diff) / (math.pi * diff)
        # Hamming window
        h[k] *= 0.54 - 0.46 * math.cos(2.0 * math.pi * k / (taps - 1))

    # Generate signal
    s = [math.sin(i * 0.01) + 0.5 * math.sin(i * 0.1) for i in range(n)]

    # Apply FIR filter
    total = 0.0
    for i in range(mid, n - mid):
        acc = 0.0
        for k in range(taps):
            acc += h[k] * s[i - mid + k]
        total += acc

    return total
