"""
Compute exponentially weighted moving average of a deterministic sequence.

Keywords: ewma, exponential, moving average, smoothing, numerical, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(500000,))
def ewma(n: int) -> list:
    """Compute the exponentially weighted moving average with alpha=0.1.

    Generates sequence as value[i] = (i * 7 + 3) % 1000 / 10.0 for determinism,
    then computes EWMA where result[0] = value[0] and
    result[i] = alpha * value[i] + (1 - alpha) * result[i-1].

    Args:
        n: Length of the sequence.

    Returns:
        List of floats representing the EWMA at each position.
    """
    alpha = 0.1
    result = []

    value = ((0 * 7 + 3) % 1000) / 10.0
    avg = value
    result.append(avg)

    for i in range(1, n):
        value = ((i * 7 + 3) % 1000) / 10.0
        avg = alpha * value + (1.0 - alpha) * avg
        result.append(avg)

    return result
