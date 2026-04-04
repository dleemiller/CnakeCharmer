"""Count valid parentheses strings among deterministically generated strings.

Keywords: leetcode, valid parentheses, stack, string, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(500000,))
def valid_parentheses(n: int) -> int:
    """Count how many of n generated bracket sequences are valid.

    For each i in range(n), generate a sequence of length 8 where
    position j is '(' if bit j of ((i * 2654435761) >> 4) is 0, else ')'.
    A sequence is valid if depth never goes negative and ends at zero.

    Args:
        n: Number of sequences to check.

    Returns:
        Count of valid parentheses sequences.
    """
    count = 0
    for i in range(n):
        bits = ((i * 2654435761) >> 4) & 0xFF
        depth = 0
        valid = True
        for j in range(8):
            if (bits >> j) & 1 == 0:
                depth += 1
            else:
                depth -= 1
            if depth < 0:
                valid = False
                break
        if valid and depth == 0:
            count += 1
    return count
