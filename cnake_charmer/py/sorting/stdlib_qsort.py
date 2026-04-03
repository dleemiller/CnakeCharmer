"""Sort integers using C stdlib qsort-style comparison callback.

Demonstrates calling convention for C qsort: generate n deterministic
integers, sort them using a comparison function, and return a checksum.

Keywords: sorting, qsort, stdlib, callback, comparison, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(300000,))
def stdlib_qsort(n: int) -> tuple:
    """Sort n deterministic integers using Python sorted() and return checksum.

    Generates arr[i] = ((i * 2654435761) ^ (i * 40503)) % 1000000 for
    determinism, sorts ascending, then computes a checksum from the first
    and last 10 elements plus the total sum.

    Args:
        n: Number of integers to sort.

    Returns:
        Tuple of (sum_first_10, sum_last_10, total_sum).
    """
    arr = [((i * 2654435761) ^ (i * 40503)) % 1000000 for i in range(n)]

    # In-place sort using comparison (mirrors C qsort behaviour)
    arr.sort()

    limit = min(10, n)
    sum_first = 0
    for i in range(limit):
        sum_first += arr[i]

    sum_last = 0
    for i in range(n - limit, n):
        sum_last += arr[i]

    total = 0
    for v in arr:
        total += v

    return (sum_first, sum_last, total)
