"""Find longest common substring of two deterministic strings.

Keywords: string processing, longest common substring, dynamic programming, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(2000,))
def longest_common_substring_rolling(n: int) -> tuple:
    """Find longest common substring using DP with rolling row.

    Strings are generated using xorshift PRNG with seeds 12345 and 67890
    over a 4-symbol alphabet for meaningful overlap.
    Uses O(n) space rolling-row DP.

    Args:
        n: Length of each string.

    Returns:
        Tuple of (length, start_pos_a, start_pos_b).
    """
    # Build strings using xorshift PRNG with small alphabet
    sa = [0] * n
    sb = [0] * n
    s = 12345
    for i in range(n):
        s ^= (s << 13) & 0xFFFFFFFF
        s ^= (s >> 17) & 0xFFFFFFFF
        s ^= (s << 5) & 0xFFFFFFFF
        sa[i] = s % 4
    s = 67890
    for i in range(n):
        s ^= (s << 13) & 0xFFFFFFFF
        s ^= (s >> 17) & 0xFFFFFFFF
        s ^= (s << 5) & 0xFFFFFFFF
        sb[i] = s % 4

    max_len = 0
    start_a = 0
    start_b = 0

    # Rolling row DP
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)

    for i in range(1, n + 1):
        for j in range(1, n + 1):
            if sa[i - 1] == sb[j - 1]:
                curr[j] = prev[j - 1] + 1
                if curr[j] > max_len:
                    max_len = curr[j]
                    start_a = i - max_len
                    start_b = j - max_len
            else:
                curr[j] = 0
        # Swap rows
        prev, curr = curr, prev
        # Clear curr for next iteration
        for j in range(n + 1):
            curr[j] = 0

    return (max_len, start_a, start_b)
