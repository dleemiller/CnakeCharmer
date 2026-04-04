"""Count distinct k-grams using rolling hash (Rabin-Karp style).

Keywords: string processing, rolling hash, k-gram, distinct substrings, benchmark
"""

from cnake_data.benchmarks import python_benchmark

BASE = 131
MOD = 10**9 + 7
K = 8


@python_benchmark(args=(100000,))
def rolling_hash_distinct(n: int) -> tuple:
    """Count distinct k-grams (k=8) in a string of length n over alphabet ACGT.

    String generation: char[i] = "ACGT"[(i*1664525+1013904223)>>30 & 3]
    Rolling hash: h = (h * BASE - char[i-k] * BASE^k + char[i]) % MOD

    After computing all hashes, sort them and count distinct values.

    Args:
        n: Length of the string.

    Returns:
        Tuple of (num_distinct, min_hash, max_hash).
    """
    if n < K:
        return (0, 0, 0)

    # Generate sequence as integers (ordinal values of ACGT chars: 65, 67, 71, 84)
    alphabet = [65, 67, 71, 84]  # ord('A'), ord('C'), ord('G'), ord('T')
    seq = [alphabet[((i * 1664525 + 1013904223) >> 30) & 3] for i in range(n)]

    # Compute BASE^K % MOD
    base_k = 1
    for _ in range(K):
        base_k = (base_k * BASE) % MOD

    # Compute initial hash for first k-gram
    h = 0
    for i in range(K):
        h = (h * BASE + seq[i]) % MOD

    hashes = [0] * (n - K + 1)
    hashes[0] = h

    # Rolling update
    for i in range(1, n - K + 1):
        h = (h * BASE - seq[i - 1] * base_k + seq[i + K - 1]) % MOD
        if h < 0:
            h += MOD
        hashes[i] = h

    hashes.sort()

    num_distinct = 0
    min_hash = hashes[0]
    max_hash = hashes[-1]
    prev = -1
    for hv in hashes:
        if hv != prev:
            num_distinct += 1
            prev = hv

    return (num_distinct, min_hash, max_hash)
