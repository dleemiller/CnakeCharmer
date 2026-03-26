# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Simulate bloom filter with multiple hash functions, measure false positive rate.

Keywords: algorithms, bloom filter, hashing, probabilistic, false positive, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from libc.string cimport memset
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(200000,))
def bloom_filter(int n):
    """Simulate a bloom filter and count false positives."""
    cdef int k = 5
    cdef int m = 8 * n

    cdef char *bits = <char *>malloc(m * sizeof(char))
    if not bits:
        raise MemoryError()
    memset(bits, 0, m * sizeof(char))

    cdef unsigned int h1, h2, val_u
    cdef int i, j, pos, val
    cdef int hashes[5]
    cdef int found, in_set
    cdef int true_positives = 0
    cdef int false_positives = 0
    cdef int bits_set = 0

    # Insert items: values 0, 3, 6, ..., 3*(n-1)
    for i in range(n):
        val = i * 3
        val_u = <unsigned int>val
        h1 = (val_u * <unsigned int>2654435761) ^ (val_u >> 16)
        h2 = (val_u * <unsigned int>2246822519) ^ (val_u >> 13)
        for j in range(k):
            pos = (h1 + j * h2) % m
            bits[pos] = 1

    # Count bits set
    for i in range(m):
        bits_set += bits[i]

    # Probe: test values 0..2*n-1
    for i in range(2 * n):
        val_u = <unsigned int>i
        h1 = (val_u * <unsigned int>2654435761) ^ (val_u >> 16)
        h2 = (val_u * <unsigned int>2246822519) ^ (val_u >> 13)

        found = 1
        for j in range(k):
            pos = (h1 + j * h2) % m
            if bits[pos] == 0:
                found = 0
                break

        in_set = (i % 3 == 0 and i < 3 * n)
        if found:
            if in_set:
                true_positives += 1
            else:
                false_positives += 1

    free(bits)
    return (true_positives, false_positives, bits_set)
