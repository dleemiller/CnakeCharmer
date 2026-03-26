"""Shell sort with Knuth gap sequence.

Keywords: sorting, shell sort, knuth gaps, algorithm, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(100000,))
def shell_sort(n: int) -> list[int]:
    """Sort a deterministic array using Shell sort with Knuth gaps.

    Generates arr[i] = (i * 31 + 17) % n, then sorts using Shell sort
    with Knuth's gap sequence (1, 4, 13, 40, 121, ...).

    Args:
        n: Number of elements to sort.

    Returns:
        The sorted list.
    """
    arr = [(i * 31 + 17) % n for i in range(n)]

    # Compute Knuth gaps: h = 3*h + 1
    gap = 1
    gaps = []
    while gap < n:
        gaps.append(gap)
        gap = 3 * gap + 1
    gaps.reverse()

    for gap in gaps:
        for i in range(gap, n):
            temp = arr[i]
            j = i
            while j >= gap and arr[j - gap] > temp:
                arr[j] = arr[j - gap]
                j -= gap
            arr[j] = temp

    return arr
