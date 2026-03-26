"""
Count total palindromic substrings using Manacher's algorithm.

Keywords: string processing, palindrome, manacher, algorithm, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(500000,))
def manacher(n: int) -> int:
    """Count total palindromic substrings using Manacher's algorithm.

    String: s[i] = chr(65 + (i * 7 + 3) % 26).

    Args:
        n: Length of the string.

    Returns:
        Total number of palindromic substrings.
    """
    if n <= 0:
        return 0

    # Build the string
    s = [chr(65 + (i * 7 + 3) % 26) for i in range(n)]

    # Transform: insert '#' between chars and at ends
    # "#a#b#c#" allows uniform handling of odd/even palindromes
    t_len = 2 * n + 1
    t = ["#"] * t_len
    for i in range(n):
        t[2 * i + 1] = s[i]

    # Manacher's algorithm
    p = [0] * t_len
    center = 0
    right = 0

    for i in range(t_len):
        mirror = 2 * center - i
        if i < right:
            p[i] = min(right - i, p[mirror])

        # Expand
        a = i + p[i] + 1
        b = i - p[i] - 1
        while a < t_len and b >= 0 and t[a] == t[b]:
            p[i] += 1
            a += 1
            b -= 1

        if i + p[i] > right:
            center = i
            right = i + p[i]

    # Count palindromic substrings
    # Each p[i] at an original character position contributes (p[i]+1)//2 odd palindromes
    # Each p[i] at a '#' position contributes p[i]//2 even palindromes
    total = 0
    for i in range(t_len):
        # Number of palindromic substrings centered here
        total += (p[i] + 1) // 2

    return total
