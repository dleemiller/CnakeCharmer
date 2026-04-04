"""Longest increasing subsequence with intermediate state tracking.

Keywords: dynamic programming, LIS, longest increasing subsequence, patience sort, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(20000,))
def longest_increasing_subseq(n: int) -> tuple:
    """Compute LIS length and intermediate state on deterministic sequence.

    Sequence: seq[i] = (i * 2654435761) % (n * 3) for i in range(n).
    Uses O(n^2) DP to also track dp values.

    Args:
        n: Length of the sequence.

    Returns:
        Tuple of (lis_length, dp_mid_val, tail_array_sum).
    """
    # Generate deterministic sequence
    mod = n * 3
    seq = [0] * n
    for i in range(n):
        seq[i] = (i * 2654435761) % mod

    # O(n log n) patience sorting with tails array
    tails = [0] * n
    tails_len = 0

    # Also compute dp[i] = LIS ending at i using binary search
    dp = [0] * n

    for i in range(n):
        val = seq[i]
        lo = 0
        hi = tails_len
        while lo < hi:
            mid = (lo + hi) // 2
            if tails[mid] < val:
                lo = mid + 1
            else:
                hi = mid
        tails[lo] = val
        dp[i] = lo + 1
        if lo == tails_len:
            tails_len += 1

    lis_length = tails_len
    dp_mid_val = dp[n // 2]
    tail_array_sum = 0
    for i in range(tails_len):
        tail_array_sum += tails[i]

    return (lis_length, dp_mid_val, tail_array_sum)
