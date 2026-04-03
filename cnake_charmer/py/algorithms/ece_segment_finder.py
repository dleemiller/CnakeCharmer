"""Enriched-region segment finder using prefix sums.

Finds all Enriched Contiguous Elements (ECE) in a scored sequence
using prefix-sum arrays and min/max sweeps. Returns (start, end)
pairs of qualifying segments that exceed a minimum length.

Keywords: algorithms, prefix sum, segment, subsequence, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(500,))
def ece_segment_finder(n: int) -> list:
    """Find all ECE segments in a synthetic scored sequence.

    Builds a sequence of +1/-4 scores from a deterministic pattern,
    then uses prefix sums with min/max sweep to identify enriched
    contiguous segments of at least min_length.

    Args:
        n: Length of the scored sequence (controls problem size).

    Returns:
        List of (start, end) tuples for each qualifying ECE segment.
    """
    # Build deterministic scored sequence: +1 if (i*7 % 5 != 0) else -4
    s = [0]  # leading zero required by algorithm
    for i in range(1, n + 1):
        if (i * 7) % 5 != 0:
            s.append(1)
        else:
            s.append(-4)

    L = len(s)
    min_ece_length = max(5, n // 20)

    # Prefix sums
    r = [0.0] * L
    for i in range(1, L):
        r[i] = r[i - 1] + s[i]

    # Running minimum from left
    X = [0.0] * L
    for i in range(1, L):
        X[i] = min(X[i - 1], r[i])

    # Running maximum from right
    Y = [0.0] * L
    Y[L - 1] = r[L - 1]
    for i in range(L - 2, -1, -1):
        Y[i] = max(Y[i + 1], r[i])

    # Sweep to find ECE segments
    bests = []
    i = 0
    j = 0
    while j < L:
        if j == L - 1 or Y[j + 1] < X[i]:
            if j - i >= min_ece_length:
                bests.append((i + 1, j))
            j += 1
            while j < L and i < L and Y[j] < X[i]:
                i += 1
        else:
            j += 1

    return bests
