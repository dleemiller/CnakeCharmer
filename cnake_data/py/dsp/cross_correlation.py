"""Cross-correlation of two signals for n lags.

Computes the cross-correlation using direct O(n^2) summation and returns
the maximum value.

Keywords: dsp, cross-correlation, correlation, lag, signal, benchmark
"""

import math

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(3000,))
def cross_correlation(n: int) -> float:
    """Compute cross-correlation and return the maximum value.

    Signals: x[i] = sin(i*0.1), y[i] = sin(i*0.1 + 0.5).

    Args:
        n: Number of samples and lags.

    Returns:
        Maximum cross-correlation value.
    """
    x = [math.sin(i * 0.1) for i in range(n)]
    y = [math.sin(i * 0.1 + 0.5) for i in range(n)]

    max_corr = -1e300
    for lag in range(n):
        acc = 0.0
        for i in range(n - lag):
            acc += x[i] * y[i + lag]
        if acc > max_corr:
            max_corr = acc

    return max_corr
