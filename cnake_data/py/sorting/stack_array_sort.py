"""Sort fixed-size blocks with insertion sort using stack-allocated array.

Keywords: sorting, insertion sort, stack array, fixed-size, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(100000,))
def stack_array_sort(n: int) -> int:
    """Sort 1024-element blocks with insertion sort, return checksum.

    Args:
        n: Controls number of iterations (n // 1024 blocks).

    Returns:
        Checksum of all sorted blocks.
    """
    iterations = n // 1024
    if iterations < 1:
        iterations = 1
    checksum = 0

    for it in range(iterations):
        arr = [0] * 1024
        seed = it * 2654435761 + 17
        for i in range(1024):
            seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
            arr[i] = seed % 100000

        # Insertion sort
        for i in range(1, 1024):
            key = arr[i]
            j = i - 1
            while j >= 0 and arr[j] > key:
                arr[j + 1] = arr[j]
                j -= 1
            arr[j + 1] = key

        for i in range(0, 1024, 64):
            checksum += arr[i]

    return checksum
