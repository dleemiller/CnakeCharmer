"""Sort blocks of elements with GIL release, returning checksum.

Keywords: sorting, insertion sort, nogil, blocks, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(1000,))
def nogil_sort_blocks(n: int) -> int:
    """Sort n blocks of 256 elements each, return checksum.

    Each block is filled with deterministic hash-based values.
    After sorting all blocks, a checksum is computed from
    selected elements.

    Args:
        n: Number of blocks to sort.

    Returns:
        Integer checksum over sorted blocks.
    """
    block_size = 256
    checksum = 0

    for b in range(n):
        block = [0] * block_size
        for i in range(block_size):
            v = (b * 7919 + i * 104729 + 31) & 0x7FFFFFFF
            v = ((v ^ (v >> 16)) * 73244475) & 0x7FFFFFFF
            block[i] = v

        # Insertion sort
        for i in range(1, block_size):
            key = block[i]
            j = i - 1
            while j >= 0 and block[j] > key:
                block[j + 1] = block[j]
                j -= 1
            block[j + 1] = key

        # Checksum from first, mid, last elements
        checksum += block[0] + block[block_size // 2] + block[-1]
        checksum &= 0x7FFFFFFFFFFFFFFF

    return checksum
