"""
Z-function computation for a string.

Keywords: string processing, z-function, pattern matching, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(500000,))
def z_function(n: int) -> int:
    """Compute the Z-function of a deterministic string and return the sum of all Z-values.

    String: s[i] = chr(65 + (i*7+3) % 26) for i in range(n).
    Z[i] = length of longest substring starting at i that matches a prefix of s.
    Z[0] = 0 by convention.

    Uses the standard O(n) Z-algorithm.

    Args:
        n: Length of the string.

    Returns:
        Sum of all Z-values.
    """
    s = [chr(65 + (i * 7 + 3) % 26) for i in range(n)]

    z = [0] * n
    left = 0
    r = 0
    for i in range(1, n):
        if i < r:
            z[i] = min(r - i, z[i - left])
        while i + z[i] < n and s[z[i]] == s[i + z[i]]:
            z[i] += 1
        if i + z[i] > r:
            left = i
            r = i + z[i]

    total = 0
    for i in range(n):
        total += z[i]

    return total
