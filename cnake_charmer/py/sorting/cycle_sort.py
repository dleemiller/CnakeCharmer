"""Cycle sort algorithm (optimal writes).

Keywords: sorting, cycle sort, minimal writes, in-place, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(3000,))
def cycle_sort(n: int) -> int:
    """Count the number of writes performed by cycle sort.

    Generates arr[i] = (i * 31 + 17) % n, then sorts using cycle sort,
    which minimizes the number of writes to the array.

    Args:
        n: Number of elements to sort.

    Returns:
        The total number of writes performed.
    """
    arr = [(i * 31 + 17) % n for i in range(n)]

    writes = 0

    for cycle_start in range(n - 1):
        item = arr[cycle_start]

        # Find the position where item should go
        pos = cycle_start
        for i in range(cycle_start + 1, n):
            if arr[i] < item:
                pos += 1

        # If item is already in correct position, skip
        if pos == cycle_start:
            continue

        # Skip duplicates
        while item == arr[pos]:
            pos += 1

        # Put the item in its correct position
        arr[pos], item = item, arr[pos]
        writes += 1

        # Rotate the rest of the cycle
        while pos != cycle_start:
            pos = cycle_start
            for i in range(cycle_start + 1, n):
                if arr[i] < item:
                    pos += 1

            while item == arr[pos]:
                pos += 1

            arr[pos], item = item, arr[pos]
            writes += 1

    return writes
