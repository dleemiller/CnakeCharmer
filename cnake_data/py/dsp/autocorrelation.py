"""Compute autocorrelation of a deterministic signal for lags 0..99.

Returns discriminating tuple of autocorrelation values.

Keywords: dsp, autocorrelation, correlation, lag, signal, benchmark
"""

import math

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(100000,))
def autocorrelation(n: int) -> tuple:
    """Compute autocorrelation for lags 0..99 and return key values.

    Signal: s[i] = sin(i*0.1) * cos(i*0.03).

    Args:
        n: Signal length.

    Returns:
        Tuple of (r0, r_mid, r_last) where r0 is autocorrelation at lag 0,
        r_mid at lag 50, r_last at lag 99.
    """
    s = [math.sin(i * 0.1) * math.cos(i * 0.03) for i in range(n)]

    max_lag = 100
    r = [0.0] * max_lag
    for lag in range(max_lag):
        acc = 0.0
        for i in range(n - lag):
            acc += s[i] * s[i + lag]
        r[lag] = acc

    return (r[0], r[max_lag // 2], r[max_lag - 1])
