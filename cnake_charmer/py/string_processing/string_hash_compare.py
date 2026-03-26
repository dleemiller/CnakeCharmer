"""
Count matching pairs of rolling hashes among n strings of length 8.

Keywords: string processing, hashing, rolling hash, comparison, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(200000,))
def string_hash_compare(n: int) -> int:
    """Count pairs of strings with matching hashes.

    Generates n strings of length 8: s[i][j] = chr(65 + (i*j + 3) % 26).
    Computes a polynomial rolling hash for each string, then counts
    how many pairs share the same hash using a frequency table.

    Args:
        n: Number of strings to generate.

    Returns:
        Number of pairs with matching hashes.
    """
    BASE = 31
    MOD = 1000000007

    # Compute hash for each string
    freq = {}
    for i in range(n):
        h = 0
        for j in range(8):
            ch = (i * j + 3) % 26
            h = (h * BASE + ch) % MOD
        if h in freq:
            freq[h] += 1
        else:
            freq[h] = 1

    # Count pairs: for each bucket with k strings, pairs = k*(k-1)/2
    total = 0
    for count in freq.values():
        total += count * (count - 1) // 2

    return total
