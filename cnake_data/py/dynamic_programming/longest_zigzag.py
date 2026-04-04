"""
Length of longest zigzag subsequence in a deterministic sequence.

Keywords: dynamic programming, zigzag, subsequence, alternating, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(5000000,))
def longest_zigzag(n: int) -> int:
    """Find length of longest zigzag (alternating) subsequence.

    A zigzag subsequence alternates between increasing and decreasing.
    Sequence: v[i] = (i * 31 + 17) % 1000.

    Args:
        n: Length of the sequence.

    Returns:
        Length of the longest zigzag subsequence.
    """
    if n <= 0:
        return 0
    if n == 1:
        return 1

    # up[i] = length of longest zigzag ending at i with last move up
    # down[i] = length ending at i with last move down
    # Optimized O(n) approach: track running up/down lengths
    up = 1
    down = 1

    for i in range(1, n):
        vi = (i * 31 + 17) % 1000
        vi_prev = ((i - 1) * 31 + 17) % 1000
        if vi > vi_prev:
            up = down + 1
        elif vi < vi_prev:
            down = up + 1

    return max(up, down)
