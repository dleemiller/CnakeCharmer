"""Clear every other block of 32 elements with zeroing, return sum.

Keywords: algorithms, memset, clear, pattern, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(100000,))
def memset_clear_pattern(n: int) -> int:
    """Generate array, zero every other 32-element block, return sum.

    Args:
        n: Number of elements.

    Returns:
        Sum of array after clearing.
    """
    arr = [0] * n
    for i in range(n):
        arr[i] = ((i * 2654435761 + 17) ^ (i * 1664525)) & 0xFFFF

    block = 32
    num_blocks = n // block
    for b in range(0, num_blocks, 2):
        start = b * block
        for j in range(block):
            arr[start + j] = 0

    total = 0
    for i in range(n):
        total += arr[i]

    return total
