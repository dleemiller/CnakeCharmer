# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Count distinct k-grams using rolling hash (Cython-optimized).

Keywords: string processing, rolling hash, k-gram, distinct substrings, cython, benchmark
"""

from libc.stdlib cimport malloc, free, qsort
from cnake_charmer.benchmarks import cython_benchmark


cdef int _cmp_ll(const void *a, const void *b) noexcept nogil:
    cdef long long va = (<long long *>a)[0]
    cdef long long vb = (<long long *>b)[0]
    if va < vb:
        return -1
    if va > vb:
        return 1
    return 0


@cython_benchmark(syntax="cy", args=(100000,))
def rolling_hash_distinct(int n):
    """Count distinct k-grams (k=8) in a string of length n over alphabet ACGT.

    Args:
        n: Length of the string.

    Returns:
        Tuple of (num_distinct, min_hash, max_hash).
    """
    cdef int K = 8
    cdef long long BASE = 131
    cdef long long MOD = 1000000007

    if n < K:
        return (0, 0, 0)

    cdef int num_hashes = n - K + 1
    cdef int *seq = <int *>malloc(n * sizeof(int))
    cdef long long *hashes = <long long *>malloc(num_hashes * sizeof(long long))
    if not seq or not hashes:
        free(seq)
        free(hashes)
        raise MemoryError()

    cdef int alphabet[4]
    alphabet[0] = 65   # A
    alphabet[1] = 67   # C
    alphabet[2] = 71   # G
    alphabet[3] = 84   # T

    cdef int i
    cdef long long h, base_k
    cdef int num_distinct
    cdef long long prev, min_hash, max_hash
    cdef unsigned int lcg

    # Generate sequence
    with nogil:
        for i in range(n):
            lcg = (<unsigned int>i) * <unsigned int>1664525 + <unsigned int>1013904223
            seq[i] = alphabet[(lcg >> 30) & 3]

    # Compute BASE^K % MOD
    base_k = 1
    with nogil:
        for i in range(K):
            base_k = (base_k * BASE) % MOD

    # Compute initial hash for first k-gram
    h = 0
    with nogil:
        for i in range(K):
            h = (h * BASE + seq[i]) % MOD
        hashes[0] = h

        # Rolling updates
        for i in range(1, num_hashes):
            h = (h * BASE - seq[i - 1] * base_k + seq[i + K - 1]) % MOD
            if h < 0:
                h = h + MOD
            hashes[i] = h

    # Sort using qsort
    qsort(hashes, num_hashes, sizeof(long long), _cmp_ll)

    # Count distinct and find min/max
    with nogil:
        min_hash = hashes[0]
        max_hash = hashes[num_hashes - 1]
        num_distinct = 0
        prev = -1
        for i in range(num_hashes):
            if hashes[i] != prev:
                num_distinct += 1
                prev = hashes[i]

    free(seq)
    free(hashes)
    return (num_distinct, min_hash, max_hash)
