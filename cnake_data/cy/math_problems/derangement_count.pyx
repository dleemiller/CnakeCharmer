# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""
Batch derangement (subfactorial) computation for k=1..n.

A derangement of k elements is a permutation where no element appears
in its original position. D(0)=1, D(1)=0, D(k)=(k-1)*(D(k-1)+D(k-2)).

Keywords: math, derangement, subfactorial, permutation, combinatorics, cython, benchmark
"""

from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(500000,))
def derangement_count(int n):
    """Compute derangement numbers D(k) mod p for k=1..n and return summary stats."""
    cdef long long MOD = 1000000007
    cdef long long total = 0
    cdef long long count_even = 0
    cdef long long prev2 = 1   # D(0)
    cdef long long prev1 = 0   # D(1)
    cdef long long d_k = 0
    cdef int k

    # k=1: D(1)=0, even
    total = 0
    count_even = 1

    with nogil:
        for k in range(2, n + 1):
            d_k = ((k - 1) * ((prev1 + prev2) % MOD)) % MOD
            total = (total + d_k) % MOD
            if d_k % 2 == 0:
                count_even = count_even + 1
            prev2 = prev1
            prev1 = d_k

    return (total, count_even)
