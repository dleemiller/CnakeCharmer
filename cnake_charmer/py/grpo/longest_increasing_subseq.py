"""Length of longest increasing subsequence via patience sorting.

Keywords: grpo, dynamic programming, subsequence, binary search, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(50000,))
def longest_increasing_subseq(n: int) -> tuple:
    """Find LIS length using patience sorting (O(n log n)).

    Generates a deterministic sequence and finds the longest strictly
    increasing subsequence using binary search on tail values.

    Returns (lis_length, number of binary searches performed, final tails checksum).

    Args:
        n: Length of the input sequence.

    Returns:
        Tuple of (lis_length, searches, checksum).
    """
    # Generate deterministic sequence
    seq = [0] * n
    for i in range(n):
        seq[i] = (i * 2654435761) & 0xFFFFFFFF  # Knuth multiplicative hash

    # Patience sorting with binary search
    tails = []
    searches = 0

    for i in range(n):
        val = seq[i]
        # Binary search for leftmost tail >= val
        lo = 0
        hi = len(tails)
        while lo < hi:
            mid = (lo + hi) >> 1
            if tails[mid] < val:
                lo = mid + 1
            else:
                hi = mid
            searches += 1

        if lo == len(tails):
            tails.append(val)
        else:
            tails[lo] = val

    # Checksum
    checksum = 0
    for v in tails:
        checksum = (checksum + v) & 0xFFFFFFFF

    return (len(tails), searches, checksum)
