"""Compute autocorrelation of a signal for lags 0..99.

Returns the sum of autocorrelation values across all computed lags.

Keywords: dsp, autocorrelation, correlation, lag, signal, benchmark
"""

import math

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(100000,))
def autocorrelation(n: int) -> float:
    """Compute autocorrelation for lags 0..99 and return sum.

    Signal: s[i] = sin(i*0.1) * cos(i*0.03).

    Args:
        n: Signal length.

    Returns:
        Sum of autocorrelation values for lags 0..99.
    """
    s = [math.sin(i * 0.1) * math.cos(i * 0.03) for i in range(n)]

    max_lag = 100
    total = 0.0
    for lag in range(max_lag):
        acc = 0.0
        for i in range(n - lag):
            acc += s[i] * s[i + lag]
        total += acc

    return total
