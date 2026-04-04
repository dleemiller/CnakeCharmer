"""
Levenshtein edit distance.

Keywords: string processing, edit distance, levenshtein, dynamic programming, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(500,))
def edit_distance(n: int) -> int:
    """Compute Levenshtein edit distance between two deterministic strings.

    Args:
        n: Half-length parameter. s1 = "ab" * n, s2 = "ba" * n.

    Returns:
        The edit distance as an integer.
    """
    s1 = "ab" * n
    s2 = "ba" * n
    len1 = len(s1)
    len2 = len(s2)

    # Use two rows for space efficiency
    prev = list(range(len2 + 1))
    curr = [0] * (len2 + 1)

    for i in range(1, len1 + 1):
        curr[0] = i
        for j in range(1, len2 + 1):
            if s1[i - 1] == s2[j - 1]:
                curr[j] = prev[j - 1]
            else:
                curr[j] = 1 + min(prev[j], curr[j - 1], prev[j - 1])
        prev, curr = curr, prev

    return (prev[len2], prev[len2 // 2])
