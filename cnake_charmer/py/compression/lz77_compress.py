"""LZ77-style compression match counting.

Keywords: compression, lz77, string matching, sliding window, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(100000,))
def lz77_compress(n: int) -> int:
    """Count LZ77-style (offset, length) match pairs in a deterministic string.

    Generates s[i] = chr(65 + (i * 7 + 3) % 26), then scans left-to-right
    with a sliding window of max size 100. For each position, finds the
    longest match in the window. Counts how many positions have a match
    of length >= 2.

    Args:
        n: Length of the string.

    Returns:
        Number of match pairs found (positions with match length >= 2).
    """
    s = [chr(65 + (i * 7 + 3) % 26) for i in range(n)]

    match_count = 0
    max_window = 100
    i = 0

    while i < n:
        best_length = 0
        window_start = max(0, i - max_window)

        for j in range(window_start, i):
            length = 0
            while i + length < n and s[j + length] == s[i + length] and j + length < i:
                length += 1
            if length > best_length:
                best_length = length

        if best_length >= 2:
            match_count += 1
            i += best_length
        else:
            i += 1

    return match_count
