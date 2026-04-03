"""
Batch derangement (subfactorial) computation for k=1..n.

A derangement of k elements is a permutation where no element appears
in its original position. D(0)=1, D(1)=0, D(k)=(k-1)*(D(k-1)+D(k-2)).

Keywords: math, derangement, subfactorial, permutation, combinatorics, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(500000,))
def derangement_count(n: int) -> tuple:
    """Compute derangement numbers D(k) mod p for k=1..n and return summary stats.

    Uses the recurrence D(k) = (k-1) * (D(k-1) + D(k-2)) with modular
    arithmetic to avoid overflow.

    Args:
        n: Upper bound (inclusive) for derangement computation.

    Returns:
        Tuple of (sum_of_derangements_mod_p, count_even_derangements).
    """
    MOD = 1000000007
    total = 0
    count_even = 0

    # D(0) = 1, D(1) = 0
    prev2 = 1  # D(k-2), starting at D(0)
    prev1 = 0  # D(k-1), starting at D(1)

    # k=1: D(1)=0, which is even
    d_k = 0
    total = (total + d_k) % MOD
    count_even = count_even + 1

    k = 2
    while k <= n:
        d_k = ((k - 1) * ((prev1 + prev2) % MOD)) % MOD
        total = (total + d_k) % MOD
        if d_k % 2 == 0:
            count_even = count_even + 1
        prev2 = prev1
        prev1 = d_k
        k = k + 1

    return (total, count_even)
