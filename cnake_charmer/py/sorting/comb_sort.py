"""Comb sort algorithm.

Keywords: sorting, comb sort, gap, shrink factor, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(500000,))
def comb_sort(n: int) -> list[int]:
    """Sort a deterministic array using comb sort.

    Generates arr[i] = (i * 31 + 17) % n, then sorts using comb sort
    with a gap shrink factor of 1.3.

    Args:
        n: Number of elements to sort.

    Returns:
        The sorted list.
    """
    arr = [(i * 31 + 17) % n for i in range(n)]

    gap = n
    shrink = 1.3
    sorted_flag = False

    while not sorted_flag:
        gap = int(gap / shrink)
        if gap <= 1:
            gap = 1
            sorted_flag = True

        for i in range(n - gap):
            if arr[i] > arr[i + gap]:
                arr[i], arr[i + gap] = arr[i + gap], arr[i]
                sorted_flag = False

    return arr
