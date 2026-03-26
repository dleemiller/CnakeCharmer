"""Simulate bloom filter with multiple hash functions, measure false positive rate.

Keywords: algorithms, bloom filter, hashing, probabilistic, false positive, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(200000,))
def bloom_filter(n: int) -> tuple:
    """Simulate a bloom filter and count false positives.

    Inserts n items into a bloom filter with k=5 hash functions and a bit array
    of size 8*n. Then tests 2*n probe values (half are in the set, half are not)
    and counts true positives and false positives.

    Args:
        n: Number of items to insert.

    Returns:
        Tuple of (true_positives, false_positives, bits_set).
    """
    k = 5
    m = 8 * n  # bit array size
    bits = [0] * m

    def _hashes(val, m, k):
        mask = 0xFFFFFFFF
        h1 = (((val * 2654435761) & mask) ^ (val >> 16)) & mask
        h2 = (((val * 2246822519) & mask) ^ (val >> 13)) & mask
        results = [0] * k
        for j in range(k):
            results[j] = ((h1 + j * h2) & mask) % m
        return results

    # Insert items: values 0, 3, 6, ..., 3*(n-1)
    for i in range(n):
        val = i * 3
        for pos in _hashes(val, m, k):
            bits[pos] = 1

    # Count bits set
    bits_set = 0
    for b in bits:
        bits_set += b

    # Probe: test values 0..2*n-1
    true_positives = 0
    false_positives = 0
    for i in range(2 * n):
        # Check membership
        found = 1
        for pos in _hashes(i, m, k):
            if bits[pos] == 0:
                found = 0
                break

        # Ground truth: item is in set if i % 3 == 0 and i < 3 * n
        in_set = i % 3 == 0 and i < 3 * n
        if found:
            if in_set:
                true_positives += 1
            else:
                false_positives += 1

    return (true_positives, false_positives, bits_set)
