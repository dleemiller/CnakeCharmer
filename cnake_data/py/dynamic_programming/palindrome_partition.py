"""Minimum cuts to partition a deterministic string into palindromes.

Keywords: palindrome, partition, dynamic programming, string, minimum cuts, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(2000,))
def palindrome_partition(n: int) -> int:
    """Compute minimum cuts to partition s into palindromes.

    String s[i] = chr(65 + (i*i+3*i+1) % 4). Uses DP with a palindrome table.

    Args:
        n: Length of the string.

    Returns:
        Minimum number of cuts.
    """
    s = [chr(65 + (i * i + 3 * i + 1) % 4) for i in range(n)]

    # is_pal[i][j] = True if s[i..j] is palindrome
    is_pal = [[False] * n for _ in range(n)]
    for i in range(n):
        is_pal[i][i] = True
    for i in range(n - 1):
        is_pal[i][i + 1] = s[i] == s[i + 1]
    for length in range(3, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            is_pal[i][j] = (s[i] == s[j]) and is_pal[i + 1][j - 1]

    # cuts[i] = min cuts for s[0..i]
    cuts = list(range(n))  # worst case: cut after every char
    for i in range(1, n):
        if is_pal[0][i]:
            cuts[i] = 0
            continue
        for j in range(1, i + 1):
            if is_pal[j][i] and cuts[j - 1] + 1 < cuts[i]:
                cuts[i] = cuts[j - 1] + 1

    return (cuts[n - 1], cuts[n // 2])
