"""
Build a Fenwick tree (Binary Indexed Tree) and compute all prefix sums.

Keywords: algorithms, fenwick tree, binary indexed tree, prefix sum, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(500000,))
def fenwick_tree(n: int) -> int:
    """Build a Fenwick tree from n values and return the total of all prefix sums.

    Values: v[i] = (i*7+3) % 100 for i in 0..n-1.
    Build the BIT, then query prefix_sum(i) for each i in 0..n-1,
    and return the sum of all prefix sums.

    Args:
        n: Number of values.

    Returns:
        Tuple of (sum of all prefix sums, prefix sum at n//2).
    """
    # Build Fenwick tree (1-indexed)
    tree = [0] * (n + 1)
    for i in range(n):
        val = (i * 7 + 3) % 100
        idx = i + 1
        while idx <= n:
            tree[idx] += val
            idx += idx & (-idx)

    # Query all prefix sums and accumulate
    total = 0
    prefix_at_n_half = 0
    n_half = n // 2
    for i in range(n):
        idx = i + 1
        s = 0
        while idx > 0:
            s += tree[idx]
            idx -= idx & (-idx)
        total += s
        if i == n_half:
            prefix_at_n_half = s

    return (total, prefix_at_n_half)
