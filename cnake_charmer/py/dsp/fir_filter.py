"""Apply a 31-tap low-pass FIR filter to a sinusoidal signal.

Coefficients use a sinc-Hamming window design. Returns discriminating
tuple of output metrics.

Keywords: dsp, FIR, filter, convolution, low-pass, Hamming, signal, benchmark
"""

import math

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(500000,))
def fir_filter(n: int) -> tuple:
    """Apply a 31-tap FIR filter and return output metrics.

    Signal: s[i] = sin(i*0.01) + 0.5*sin(i*0.1).
    Filter: sinc * Hamming window, cutoff 0.2*pi.

    Args:
        n: Signal length.

    Returns:
        Tuple of (output_sum, output_mid_val, output_last_val).
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
    out_len = n - 2 * mid
    output = [0.0] * out_len
    for i in range(out_len):
        acc = 0.0
        for k in range(taps):
            acc += h[k] * s[i + k]
        output[i] = acc

    output_sum = 0.0
    for i in range(out_len):
        output_sum += output[i]

    output_mid_val = output[out_len // 2]
    output_last_val = output[out_len - 1]

    return (output_sum, output_mid_val, output_last_val)
