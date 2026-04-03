"""Compute Levenshtein edit distance between two strings.

Keywords: grpo, string, dynamic programming, edit distance, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(2000,))
def levenshtein_distance(n: int) -> tuple:
    """Compute Levenshtein distance between two deterministic strings.

    Generates two strings of length n from a repeating alphabet pattern
    with insertions/deletions to make them differ.

    Returns (distance, number of DP cells computed, checksum of last row).

    Args:
        n: Base length of the strings.

    Returns:
        Tuple of (distance, cells_computed, row_checksum).
    """
    # Generate deterministic strings
    alpha = "abcdefghijklmnop"
    s1 = ""
    s2 = ""
    for i in range(n):
        s1 += alpha[i % 16]
        s2 += alpha[(i * 3 + 7) % 16]

    len1 = len(s1)
    len2 = len(s2)

    # DP with two rows
    prev = list(range(len2 + 1))
    curr = [0] * (len2 + 1)
    cells = 0

    for i in range(1, len1 + 1):
        curr[0] = i
        for j in range(1, len2 + 1):
            if s1[i - 1] == s2[j - 1]:
                curr[j] = prev[j - 1]
            else:
                curr[j] = 1 + min(prev[j], curr[j - 1], prev[j - 1])
            cells += 1
        prev, curr = curr, prev

    # Checksum of final row
    checksum = 0
    for v in prev:
        checksum = (checksum * 31 + v) & 0xFFFFFFFF

    return (prev[len2], cells, checksum)
