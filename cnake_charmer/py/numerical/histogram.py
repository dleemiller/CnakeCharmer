"""
Count occurrences of values in a deterministically generated sequence.

Keywords: histogram, frequency, counting, numerical, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(1000000,))
def histogram(n: int) -> list:
    """Count occurrences of values 0..99 in a deterministic sequence of length n.

    Generates sequence as value[i] = (i * 31 + 17) % 100, then counts how
    many times each value appears.

    Args:
        n: Length of the sequence to generate.

    Returns:
        List of 100 ints where result[v] is the count of value v.
    """
    counts = [0] * 100

    for i in range(n):
        value = (i * 31 + 17) % 100
        counts[value] += 1

    return counts
