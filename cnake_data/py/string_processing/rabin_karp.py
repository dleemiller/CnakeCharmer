"""Count pattern occurrences using Rabin-Karp rolling hash.

Keywords: string processing, rabin karp, rolling hash, pattern matching, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(1000000,))
def rabin_karp(n: int) -> int:
    """Count occurrences of a 5-char pattern in a deterministic text using Rabin-Karp.

    Text: chr(65 + (i*7 + 3) % 26) for i in range(n)
    Pattern: first 5 characters of the text.
    Uses a rolling hash with base 256 and modulus 101.

    Args:
        n: Length of the text.

    Returns:
        Number of pattern occurrences.
    """
    if n < 5:
        return 0

    base = 256
    mod = 1000000007
    pat_len = 5

    # Build text as list of ordinals
    text = [65 + (i * 7 + 3) % 26 for i in range(n)]
    pattern = text[:pat_len]

    # Compute base^(pat_len-1) % mod
    h = 1
    for _ in range(pat_len - 1):
        h = (h * base) % mod

    # Hash the pattern and first window
    pat_hash = 0
    win_hash = 0
    for i in range(pat_len):
        pat_hash = (pat_hash * base + pattern[i]) % mod
        win_hash = (win_hash * base + text[i]) % mod

    count = 0
    for i in range(n - pat_len + 1):
        if win_hash == pat_hash:
            # Verify character by character
            match = True
            for j in range(pat_len):
                if text[i + j] != pattern[j]:
                    match = False
                    break
            if match:
                count += 1
        # Slide the window
        if i < n - pat_len:
            win_hash = ((win_hash - text[i] * h) * base + text[i + pat_len]) % mod
            if win_hash < 0:
                win_hash += mod

    return count
