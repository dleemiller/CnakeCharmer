"""Running variance using Welford's online algorithm.

Keywords: statistics, variance, welford, online, streaming, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(5000000,))
def running_variance(n: int) -> float:
    """Compute running variance using Welford's online algorithm.

    Data: v[i] = (i*17+5) % 1000 / 10.0.
    Returns the final population variance after processing all n values.

    Args:
        n: Number of data points.

    Returns:
        Final population variance.
    """
    mean = 0.0
    m2 = 0.0
    for i in range(n):
        val = ((i * 17 + 5) % 1000) / 10.0
        delta = val - mean
        mean += delta / (i + 1)
        delta2 = val - mean
        m2 += delta * delta2

    if n < 2:
        return 0.0
    return m2 / n
