"""Dynamically grow array by doubling capacity, simulating realloc.

Keywords: algorithms, dynamic array, realloc, grow, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(100000,))
def dynamic_array_grow(n: int) -> int:
    """Push n values into a dynamic array that doubles capacity.

    Args:
        n: Number of values to push.

    Returns:
        Sum of all values in the array.
    """
    capacity = 16
    size = 0
    arr = [0] * capacity

    for i in range(n):
        val = ((i * 2654435761 + 17) ^ (i * 1103515245)) & 0x7FFFFFFF
        if size >= capacity:
            capacity *= 2
            new_arr = [0] * capacity
            for j in range(size):
                new_arr[j] = arr[j]
            arr = new_arr
        arr[size] = val
        size += 1

    total = 0
    for i in range(size):
        total += arr[i]

    return total
