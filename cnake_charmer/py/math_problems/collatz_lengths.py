"""
Compute the Collatz sequence length for each number from 1 to n.

Keywords: collatz, sequence, math, conjecture, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(10000,))
def collatz_lengths(n: int) -> list:
    """Compute the Collatz sequence length for each number 1..n.

    For each number, count the steps needed to reach 1 by repeatedly applying:
    if even, divide by 2; if odd, multiply by 3 and add 1.

    Args:
        n: Upper bound (inclusive) for numbers to compute lengths for.

    Returns:
        List of ints where result[i] is the Collatz length for number i+1.
    """
    result = []

    for start in range(1, n + 1):
        count = 0
        val = start
        while val != 1:
            val = val // 2 if val % 2 == 0 else 3 * val + 1
            count += 1
        result.append(count)

    return result
