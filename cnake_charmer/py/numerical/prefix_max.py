"""
Compute prefix maximum of a deterministically generated sequence.

Keywords: numerical, prefix maximum, running max, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(500000,))
def prefix_max(n: int) -> list:
    """Compute the prefix maximum of a generated sequence of n numbers.

    Generates sequence as value[i] = (i * 31 + 17) % 10000 for determinism,
    then computes the running maximum at each position.

    Args:
        n: Length of the sequence.

    Returns:
        List of ints representing the prefix maximum at each position.
    """
    result = []
    current_max = 0

    for i in range(n):
        value = (i * 31 + 17) % 10000
        if value > current_max:
            current_max = value
        result.append(current_max)

    return result
