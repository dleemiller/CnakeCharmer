"""
Sum of Hamming distances between consecutive byte strings.

Keywords: string processing, hamming distance, byte comparison, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(500000,))
def hamming_distance_sum(n: int) -> int:
    """Compute sum of Hamming distances between all consecutive pairs of byte strings.

    Strings: s[i] = bytes([(i*j+3)%256 for j in range(8)]), length 8 each.

    Args:
        n: Number of byte strings.

    Returns:
        Sum of Hamming distances for all consecutive pairs.
    """
    # Generate byte strings
    strings = [bytes([(i * j + 3) % 256 for j in range(8)]) for i in range(n)]

    total = 0
    max_dist = 0
    for i in range(n - 1):
        s1 = strings[i]
        s2 = strings[i + 1]
        dist = 0
        for k in range(8):
            if s1[k] != s2[k]:
                dist += 1
        total += dist
        if dist > max_dist:
            max_dist = dist

    return (total, max_dist)
