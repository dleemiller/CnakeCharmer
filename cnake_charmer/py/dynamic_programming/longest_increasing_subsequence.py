"""
Longest increasing subsequence.

Keywords: dynamic programming, longest increasing subsequence, LIS, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(2000,))
def longest_increasing_subsequence(n: int) -> int:
    """Compute the length of the longest increasing subsequence.

    Uses a deterministic sequence: seq[i] = i*7 % n for i in range(n).
    Uses patience sorting (O(n log n)) with binary search.

    Args:
        n: Length of the sequence.

    Returns:
        Length of the LIS.
    """
    seq = [i * 7 % n for i in range(n)]

    # tails[i] = smallest tail element for increasing subsequence of length i+1
    tails = []

    for val in seq:
        lo = 0
        hi = len(tails)
        while lo < hi:
            mid = (lo + hi) // 2
            if tails[mid] < val:
                lo = mid + 1
            else:
                hi = mid
        if lo == len(tails):
            tails.append(val)
        else:
            tails[lo] = val

    return (len(tails), tails[-1] if tails else 0)
