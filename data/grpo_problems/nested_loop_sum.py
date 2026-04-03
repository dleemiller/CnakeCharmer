def nested_loop_sum(n, batch_size):
    """Compute the total count from a nested loop iteration.

    Simulates a parallel workload pattern where an outer loop of n iterations
    each performs batch_size inner increments. The result is n * batch_size.

    Args:
        n: number of outer loop iterations
        batch_size: number of inner loop iterations per outer step

    Returns:
        The accumulated sum (an integer equal to n * batch_size).
    """
    total = 0
    for _i in range(n):
        for _j in range(batch_size):
            total += 1
    return total
