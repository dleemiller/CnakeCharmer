"""KMP pattern matching returning all match positions count and last position.

Keywords: string processing, KMP, Knuth-Morris-Pratt, pattern matching, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(2000000,))
def knuth_morris_pratt_all(n: int) -> tuple:
    """Find all occurrences of a pattern in deterministic text using KMP.

    Text: chr(97 + (i*7+3) % 26) for i in range(n).
    Pattern: first 6 characters of the text.
    Returns (match_count, last_match_position, failure_table_checksum) where
    failure_table_checksum = sum of failure[i] * (i+1) for the pattern.

    Args:
        n: Length of the text.

    Returns:
        Tuple of (match count, last match position, failure table checksum).
    """
    pat_len = 6
    if n < pat_len:
        return (0, -1, 0)

    # Build text as list of ordinals
    text = [97 + (i * 7 + 3) % 26 for i in range(n)]
    pattern = text[:pat_len]

    # Build failure table
    failure = [0] * pat_len
    k = 0
    for i in range(1, pat_len):
        while k > 0 and pattern[k] != pattern[i]:
            k = failure[k - 1]
        if pattern[k] == pattern[i]:
            k += 1
        failure[i] = k

    # Compute failure table checksum
    fail_checksum = 0
    for i in range(pat_len):
        fail_checksum += failure[i] * (i + 1)

    # KMP search
    match_count = 0
    last_pos = -1
    j = 0  # index into pattern

    for i in range(n):
        while j > 0 and pattern[j] != text[i]:
            j = failure[j - 1]
        if pattern[j] == text[i]:
            j += 1
        if j == pat_len:
            match_count += 1
            last_pos = i - pat_len + 1
            j = failure[j - 1]

    return (match_count, last_pos, fail_checksum)
