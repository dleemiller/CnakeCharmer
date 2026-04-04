"""Safe vector access with bounds checking and exception handling.

Keywords: algorithms, bounds check, exception handling, index error, vector, benchmark
"""

from cnake_data.benchmarks import python_benchmark

MOD = 10**9 + 7


@python_benchmark(args=(200000,))
def cpp_except_bounds_check(n: int) -> tuple:
    """Populate a list of size n and perform m=2*n lookups, some out of bounds.

    Values: val[i] = (i * 2654435761) % n
    Lookup indices: idx = (i * 1103515245 + 12345) % (n + n // 100)
    Some indices exceed list size. Catch IndexError, count them.

    Args:
        n: Size of the list and half the number of lookups.

    Returns:
        Tuple of (sum_of_valid_values % (10**9+7), out_of_bounds_count).
    """
    data = [(i * 2654435761) % n for i in range(n)]
    m = 2 * n
    upper = n + n // 100
    sum_valid = 0
    oob_count = 0
    for i in range(m):
        idx = (i * 1103515245 + 12345) % upper
        try:
            sum_valid = (sum_valid + data[idx]) % MOD
        except IndexError:
            oob_count += 1
    return (sum_valid, oob_count)
