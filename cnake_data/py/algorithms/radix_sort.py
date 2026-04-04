"""
Radix sort (base 10, LSD) on a deterministic integer array.

Keywords: algorithms, radix sort, sorting, LSD, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(100000,))
def radix_sort(n: int) -> list:
    """Radix sort (base 10, least-significant digit first) on a deterministic array.

    Array is generated as arr[i] = (i*31 + 17) % 100000.

    Args:
        n: Length of the array to sort.

    Returns:
        Sorted list of integers.
    """
    arr = [(i * 31 + 17) % 100000 for i in range(n)]

    max_val = 99999
    exp = 1
    while exp <= max_val:
        buckets = [[] for _ in range(10)]
        for val in arr:
            digit = (val // exp) % 10
            buckets[digit].append(val)

        arr = []
        for bucket in buckets:
            arr.extend(bucket)

        exp *= 10

    return arr
