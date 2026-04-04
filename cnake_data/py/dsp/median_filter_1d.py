"""1D median filter with configurable window size.

Keywords: dsp, median, filter, smoothing, signal processing, benchmark
"""

import math

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(50000,))
def median_filter_1d(n: int) -> tuple:
    """Apply 1D median filter with window size 7 to a deterministic signal.

    Signal: s[i] = sin(2*pi*i*3/n) + 0.5*cos(2*pi*i*7/n) + (i*17 % 53) / 53.0

    Uses insertion sort within each window to find the median.

    Args:
        n: Signal length.

    Returns:
        Tuple of (checksum, first_output, last_output).
    """
    pi2 = 2.0 * math.pi
    window = 7
    half_w = window // 2

    # Generate signal
    signal = [0.0] * n
    for i in range(n):
        signal[i] = (
            math.sin(pi2 * i * 3.0 / n) + 0.5 * math.cos(pi2 * i * 7.0 / n) + (i * 17 % 53) / 53.0
        )

    # Apply median filter
    output = [0.0] * n
    buf = [0.0] * window
    for i in range(n):
        # Gather window elements
        count = 0
        for j in range(i - half_w, i + half_w + 1):
            if 0 <= j < n:
                buf[count] = signal[j]
                count += 1

        # Insertion sort on buf[0..count-1]
        for a in range(1, count):
            key = buf[a]
            b = a - 1
            while b >= 0 and buf[b] > key:
                buf[b + 1] = buf[b]
                b -= 1
            buf[b + 1] = key

        output[i] = buf[count // 2]

    # Compute checksum
    checksum = 0.0
    for i in range(n):
        checksum += output[i] * (i + 1)

    return (checksum, output[0], output[n - 1])
