"""Cocktail shaker sort algorithm.

Keywords: sorting, cocktail sort, bidirectional bubble sort, shaker sort, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(3000,))
def cocktail_sort(n: int) -> list:
    """Sort a deterministic array using cocktail shaker sort.

    Generates arr[i] = (i * 47 + 13) % n, then sorts using
    bidirectional bubble sort (cocktail shaker sort).

    Args:
        n: Number of elements to sort.

    Returns:
        The sorted list.
    """
    arr = [None] * n
    for i in range(n):
        arr[i] = (i * 47 + 13) % n

    start = 0
    end = n - 1
    swapped = True

    while swapped:
        swapped = False

        # Forward pass
        for i in range(start, end):
            if arr[i] > arr[i + 1]:
                arr[i], arr[i + 1] = arr[i + 1], arr[i]
                swapped = True

        if not swapped:
            break

        end -= 1
        swapped = False

        # Backward pass
        for i in range(end, start, -1):
            if arr[i] < arr[i - 1]:
                arr[i], arr[i - 1] = arr[i - 1], arr[i]
                swapped = True

        start += 1

    return arr
