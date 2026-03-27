"""LZ77 compression with match chaining and token statistics.

Keywords: compression, lz77, match chain, sliding window, tokens, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(80000,))
def lz77_match_chain(n: int) -> tuple:
    """LZ77 compression with chained match search and detailed token stats.

    Generates s[i] = chr(65 + (i*11 + 5) % 20) for a 20-char alphabet,
    uses a sliding window of 128 with hash-chain lookups.
    Tokens are either literal (0) or match (offset, length).

    Args:
        n: Length of the string to compress.

    Returns:
        Tuple of (num_literal_tokens, num_match_tokens, total_match_length).
    """
    s = [0] * n
    for i in range(n):
        s[i] = (i * 11 + 5) % 20

    max_window = 128
    min_match = 3
    num_literals = 0
    num_matches = 0
    total_match_len = 0

    i = 0
    while i < n:
        best_length = 0
        window_start = i - max_window if i > max_window else 0

        # Search for longest match in window
        j = window_start
        while j < i:
            length = 0
            while i + length < n and j + length < i and s[j + length] == s[i + length]:
                length += 1
            if length > best_length:
                best_length = length
            j += 1

        if best_length >= min_match:
            num_matches += 1
            total_match_len += best_length
            i += best_length
        else:
            num_literals += 1
            i += 1

    return (num_literals, num_matches, total_match_len)
