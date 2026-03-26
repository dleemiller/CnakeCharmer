"""Apply a second-order IIR biquad filter to a sinusoidal signal.

Uses direct form I with fixed coefficients. Returns the sum of the
filtered output signal.

Keywords: dsp, IIR, biquad, filter, recursive, signal, benchmark
"""

import math

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(5000000,))
def iir_biquad(n: int) -> float:
    """Apply IIR biquad filter and return sum of filtered signal.

    Signal: s[i] = sin(i*0.05).
    Coefficients: b=[0.1, 0.2, 0.1], a=[1.0, -0.8, 0.2].

    Args:
        n: Signal length.

    Returns:
        Sum of the filtered signal.
    """
    b0, b1, b2 = 0.1, 0.2, 0.1
    a1, a2 = -0.8, 0.2

    x1 = 0.0
    x2 = 0.0
    y1 = 0.0
    y2 = 0.0
    total = 0.0

    for i in range(n):
        x0 = math.sin(i * 0.05)
        y0 = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        total += y0
        x2 = x1
        x1 = x0
        y2 = y1
        y1 = y0

    return total
