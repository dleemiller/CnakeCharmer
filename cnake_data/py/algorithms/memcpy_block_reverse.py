"""Reverse array in blocks of 64 using block copy, return checksum.

Keywords: algorithms, memcpy, block reverse, array, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(100000,))
def memcpy_block_reverse(n: int) -> int:
    """Generate n-element array, reverse in 64-element blocks.

    Args:
        n: Number of elements.

    Returns:
        Checksum of reversed array.
    """
    arr = [0] * n
    for i in range(n):
        arr[i] = ((i * 2654435761 + 17) ^ (i >> 3)) & 0x7FFFFFFF

    block = 64
    num_blocks = n // block
    for i in range(num_blocks // 2):
        lo = i * block
        hi = (num_blocks - 1 - i) * block
        # Swap blocks using temp buffer
        tmp = arr[lo : lo + block]
        for j in range(block):
            arr[lo + j] = arr[hi + j]
        for j in range(block):
            arr[hi + j] = tmp[j]

    checksum = 0
    for i in range(0, n, 128):
        checksum += arr[i]
    return checksum
