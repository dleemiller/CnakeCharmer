"""Sort integers by population count (number of 1-bits) using Python's sorted().

Keywords: sorting, popcount, bit count, custom comparator, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark

MOD = 10**9 + 7


@python_benchmark(args=(300000,))
def cpp_sort_by_popcount(n: int) -> int:
    """Generate n values, sort by (popcount, value), return position-weighted hash.

    Values: val[i] = (i * 2654435761) % (10**9)
    Sort key: (bin(x).count('1'), x)
    Hash: sum(arr[i] * (i+1)) % (10**9+7)

    Args:
        n: Number of values to generate and sort.

    Returns:
        Position-weighted hash of sorted array.
    """
    arr = [(i * 2654435761) % (10**9) for i in range(n)]
    arr.sort(key=lambda x: (bin(x).count("1"), x))
    result = 0
    for i in range(n):
        result = (result + arr[i] * (i + 1)) % MOD
    return result
