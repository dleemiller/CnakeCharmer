"""
Compute cumulative sum of a deterministically generated sequence.

Keywords: cumulative sum, prefix sum, numerical, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(500000,))
def cumulative_sum(n: int) -> list:
    """Compute the cumulative sum of a generated sequence of n numbers.

    Generates sequence as value[i] = (i * 13 + 7) % 1000 for determinism,
    then computes the prefix sum at each position.

    Args:
        n: Length of the sequence.

    Returns:
        List of ints representing the cumulative sum at each position.
    """
    result = []
    total = 0

    for i in range(n):
        value = (i * 13 + 7) % 1000
        total += value
        result.append(total)

    return result
