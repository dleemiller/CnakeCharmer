"""Exponential moving average of a deterministic signal.

Keywords: statistics, exponential moving average, EMA, signal processing, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(50000,))
def exponential_moving_average(n: int) -> tuple:
    """Compute EMA of a deterministic signal and return three sample values.

    Signal: values[i] = (i * 7919 + 12345) % 10000
    EMA weight: w = 2.0 / (window + 1), window = max(n // 10, 2)
    EMA: e[0] = values[0], e[i] = values[i] * w + e[i-1] * (1 - w)

    Args:
        n: Length of the signal.

    Returns:
        Tuple of (e[n-1], e[n//2], e[n//4]) — three independent EMA samples.
    """
    window = max(n // 10, 2)
    w = 2.0 / (window + 1)
    one_minus_w = 1.0 - w

    ema = (0 * 7919 + 12345) % 10000
    ema = float(ema)
    quarter_val = ema
    half_val = ema

    quarter = n // 4
    half = n // 2

    for i in range(1, n):
        val = float((i * 7919 + 12345) % 10000)
        ema = val * w + ema * one_minus_w
        if i == quarter:
            quarter_val = ema
        if i == half:
            half_val = ema

    return (ema, half_val, quarter_val)
