"""Count pattern occurrences in text using Knuth-Morris-Pratt algorithm.

Keywords: algorithms, string matching, KMP, pattern search, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(1000000,))
def kmp_search(n: int) -> int:
    """Count occurrences of a pattern in a deterministic text using KMP.

    Text: chr(65 + (i*7+3)%26) for i in range(n).
    Pattern: chr(65 + (i*7+3)%26) for i in range(10) (first 10 chars of text).

    Args:
        n: Length of the text.

    Returns:
        Tuple of (count of occurrences, position of last match or -1).
    """
    # Build text and pattern
    text = [65 + (i * 7 + 3) % 26 for i in range(n)]
    pattern = [65 + (i * 7 + 3) % 26 for i in range(10)]
    m = len(pattern)

    # Build KMP failure function
    fail = [0] * m
    k = 0
    for i in range(1, m):
        while k > 0 and pattern[k] != pattern[i]:
            k = fail[k - 1]
        if pattern[k] == pattern[i]:
            k += 1
        fail[i] = k

    # Search
    count = 0
    last_match_pos = -1
    k = 0
    for i in range(n):
        while k > 0 and pattern[k] != text[i]:
            k = fail[k - 1]
        if pattern[k] == text[i]:
            k += 1
        if k == m:
            count += 1
            last_match_pos = i - m + 1
            k = fail[k - 1]

    return (count, last_match_pos)
