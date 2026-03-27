"""Build a linked list and sum elements via iteration.

Keywords: linked list, iterator, sum, data structure, algorithms, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(50000,))
def linked_list_sum(n: int) -> float:
    """Build a linked list of n nodes, iterate twice computing different aggregates.

    First pass: sum all values.
    Second pass: sum squares of values.
    Returns sum + sum_of_squares.

    Args:
        n: Number of nodes.

    Returns:
        Sum of values plus sum of squared values.
    """
    # Build list as array of (value, next_index) pairs
    values = [0.0] * n
    nexts = [-1] * n

    for i in range(n):
        values[i] = ((i * 2654435761 + 13) % 10000) / 100.0
        if i < n - 1:
            nexts[i] = i + 1

    # First iteration: sum values
    total = 0.0
    idx = 0
    while idx != -1:
        total += values[idx]
        idx = nexts[idx]

    # Second iteration: sum squares
    sq_total = 0.0
    idx = 0
    while idx != -1:
        sq_total += values[idx] * values[idx]
        idx = nexts[idx]

    return total + sq_total
