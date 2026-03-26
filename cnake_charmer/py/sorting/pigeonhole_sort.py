"""Pigeonhole sort algorithm.

Keywords: sorting, pigeonhole sort, counting, distribution, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(5000000,))
def pigeonhole_sort(n: int) -> list[int]:
    """Sort a deterministic array using pigeonhole sort.

    Generates arr[i] = (i * 31 + 17) % 1000 (range 0-999), then sorts
    using pigeonhole sort.

    Args:
        n: Number of elements to sort.

    Returns:
        The sorted list.
    """
    arr = [(i * 31 + 17) % 1000 for i in range(n)]

    holes = [0] * 1000
    for i in range(n):
        holes[arr[i]] += 1

    result = []
    for i in range(1000):
        for _ in range(holes[i]):
            result.append(i)

    return result
