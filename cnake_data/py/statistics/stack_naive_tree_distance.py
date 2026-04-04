"""Compute symmetric naive tree distance over fixed-depth context tables.

Adapted from The Stack v2 Cython candidate:
- blob_id: 5da5a7e3fb722f09faaa0bb02b89936332ffdd95
- filename: naive_parameter_sampling.pyx

Keywords: statistics, tree distance, context probabilities, symmetric metric
"""

from cnake_data.benchmarks import python_benchmark


def _asym_dist(left: list, right: list, alphabet: int) -> float:
    total = 0.0
    for i, row in enumerate(left):
        clen = 1 + (i % 5)
        for c in range(alphabet):
            target = row[c]
            best = target * target
            for j, row2 in enumerate(right):
                if (1 + (j % 5)) == clen:
                    diff = row2[c] - target
                    sq = diff * diff
                    if sq < best:
                        best = sq
            total += best
    return (total / (len(left) * alphabet)) ** 0.5


@python_benchmark(args=(700, 4))
def stack_naive_tree_distance(n_contexts: int, alphabet: int) -> tuple:
    """Generate two deterministic context tables and return symmetric distance summary."""
    state = 13579
    left = []
    right = []

    for _ in range(n_contexts):
        lr = []
        rr = []
        s1 = 0.0
        s2 = 0.0
        for _c in range(alphabet):
            state = (1103515245 * state + 12345) & 0x7FFFFFFF
            v1 = 1.0 + ((state >> 8) % 100)
            state = (1103515245 * state + 12345) & 0x7FFFFFFF
            v2 = 1.0 + ((state >> 8) % 100)
            lr.append(v1)
            rr.append(v2)
            s1 += v1
            s2 += v2
        left.append([v / s1 for v in lr])
        right.append([v / s2 for v in rr])

    d1 = _asym_dist(left, right, alphabet)
    d2 = _asym_dist(right, left, alphabet)
    sym = 0.5 * (d1 + d2)

    return (int(d1 * 1_000_000), int(d2 * 1_000_000), int(sym * 1_000_000), n_contexts)
