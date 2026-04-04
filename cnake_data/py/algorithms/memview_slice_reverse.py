"""
Reverse subarrays of a deterministic array using slicing and compute a checksum.

Keywords: algorithms, reverse, slicing, subarray, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(100000,))
def memview_slice_reverse(n: int) -> float:
    """Reverse non-overlapping blocks of 64 elements, return sum of result.

    Data: data[i] = ((i * 43 + 17) % 1000) / 10.0
    Reverses each block of 64 elements in-place, then sums all elements.

    Args:
        n: Length of the array.

    Returns:
        Sum of all elements after reversals.
    """
    block = 64
    data = [0.0] * n
    for i in range(n):
        data[i] = ((i * 43 + 17) % 1000) / 10.0

    # Reverse each block of 64
    i = 0
    while i + block <= n:
        # Reverse data[i:i+block]
        left = i
        right = i + block - 1
        while left < right:
            data[left], data[right] = data[right], data[left]
            left += 1
            right -= 1
        i += block

    total = 0.0
    for i in range(n):
        total += data[i]

    return total
