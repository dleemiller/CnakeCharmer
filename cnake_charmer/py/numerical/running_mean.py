"""
Compute running mean of a deterministically generated sequence.

Keywords: running mean, cumulative average, numerical, statistics, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(100000,))
def running_mean(n: int) -> list:
    """Compute the running (cumulative) mean of a generated sequence of n numbers.

    Generates sequence as value[i] = (i * 7 + 3) % 1000 / 10.0 for determinism,
    then computes the cumulative mean at each position.

    Args:
        n: Length of the sequence.

    Returns:
        List of floats representing the running mean at each position.
    """
    result = []
    cumsum = 0.0

    for i in range(n):
        value = ((i * 7 + 3) % 1000) / 10.0
        cumsum += value
        result.append(cumsum / (i + 1))

    return result
