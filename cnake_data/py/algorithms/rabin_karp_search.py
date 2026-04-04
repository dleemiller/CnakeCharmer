"""Rabin-Karp string matching on deterministic text and pattern.

Keywords: algorithms, rabin-karp, string matching, hashing, search, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(500000,))
def rabin_karp_search(n: int) -> tuple:
    """Search for pattern occurrences in deterministic text using Rabin-Karp.

    Generates text of length n from deterministic char sequence and searches
    for a 7-character pattern using rolling hash.

    Args:
        n: Length of text to search.

    Returns:
        Tuple of (match_count, first_match_pos, last_match_pos).
    """
    # Generate deterministic text
    text = [0] * n
    for i in range(n):
        text[i] = 97 + ((i * 7 + 13) % 26)  # lowercase a-z

    # Pattern: deterministic 7-char pattern that will appear periodically
    pat_len = 7
    pattern = [0] * pat_len
    for i in range(pat_len):
        pattern[i] = 97 + ((i * 7 + 13) % 26)  # matches start of text cycle

    base = 256
    mod = 1000000007

    # Compute hash of pattern and first window
    pat_hash = 0
    win_hash = 0
    h = 1  # base^(pat_len-1) mod mod

    for _i in range(pat_len - 1):
        h = (h * base) % mod

    for i in range(pat_len):
        pat_hash = (pat_hash * base + pattern[i]) % mod
        win_hash = (win_hash * base + text[i]) % mod

    match_count = 0
    first_match = -1
    last_match = -1

    for i in range(n - pat_len + 1):
        if win_hash == pat_hash:
            # Verify match
            match = True
            for j in range(pat_len):
                if text[i + j] != pattern[j]:
                    match = False
                    break
            if match:
                match_count += 1
                if first_match == -1:
                    first_match = i
                last_match = i

        # Roll hash forward
        if i < n - pat_len:
            win_hash = (base * (win_hash - text[i] * h) + text[i + pat_len]) % mod
            if win_hash < 0:
                win_hash += mod

    return (match_count, first_match, last_match)
