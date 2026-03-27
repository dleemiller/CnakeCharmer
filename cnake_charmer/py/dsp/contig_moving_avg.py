"""
Compute a moving average with a fixed window over a deterministic signal.

Keywords: dsp, moving average, smoothing, signal processing, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(100000,))
def contig_moving_avg(n: int) -> float:
    """Compute moving average with window=7 and return the sum of the output.

    Signal: signal[i] = ((i * 61 + 29) % 1000) / 50.0

    Args:
        n: Length of the input signal.

    Returns:
        Sum of the moving average output.
    """
    window = 7
    signal = [0.0] * n
    for i in range(n):
        signal[i] = ((i * 61 + 29) % 1000) / 50.0

    out_len = n - window + 1
    total = 0.0
    inv_w = 1.0 / window

    for i in range(out_len):
        s = 0.0
        for j in range(window):
            s += signal[i + j]
        total += s * inv_w

    return total
