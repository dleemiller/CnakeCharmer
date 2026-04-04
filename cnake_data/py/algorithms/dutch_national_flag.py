"""Three-way partition (Dutch National Flag) on a deterministic array.

Keywords: algorithms, partition, dutch national flag, three-way, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(5000000,))
def dutch_national_flag(n: int) -> list:
    """Partition array of 0s, 1s, 2s using Dutch National Flag algorithm.

    Array: arr[i] = (i*31+17) % 3.

    Args:
        n: Number of elements.

    Returns:
        Sorted list of integers (all 0s, then 1s, then 2s).
    """
    arr = [(i * 31 + 17) % 3 for i in range(n)]

    lo = 0
    mid = 0
    hi = n - 1

    while mid <= hi:
        if arr[mid] == 0:
            arr[lo], arr[mid] = arr[mid], arr[lo]
            lo += 1
            mid += 1
        elif arr[mid] == 1:
            mid += 1
        else:
            arr[mid], arr[hi] = arr[hi], arr[mid]
            hi -= 1

    return arr
