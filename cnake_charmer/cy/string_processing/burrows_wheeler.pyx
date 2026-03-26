# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Burrows-Wheeler Transform and byte sum of output (Cython-optimized).

Keywords: string processing, burrows-wheeler, bwt, transform, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(5000,))
def burrows_wheeler(int n):
    """Compute BWT of a deterministic string and return sum of output bytes."""
    cdef int length = n + 1  # +1 for sentinel
    cdef char *s = <char *>malloc(length * sizeof(char))
    cdef int *sa = <int *>malloc(length * sizeof(int))
    if not s or not sa:
        if s: free(s)
        if sa: free(sa)
        raise MemoryError()

    cdef int i, j, total
    cdef int gap, idx1, idx2
    cdef char c1, c2

    # Generate string
    for i in range(n):
        s[i] = 65 + (i * 7 + 3) % 4
    s[n] = 0  # sentinel (null byte)

    # Initialize suffix array
    for i in range(length):
        sa[i] = i

    # Simple suffix array construction using shell sort
    # (adequate for n=5000 benchmarking)
    gap = length / 2
    while gap > 0:
        for i in range(gap, length):
            j = i
            while j >= gap:
                idx1 = sa[j - gap]
                idx2 = sa[j]
                # Compare suffixes
                if _compare_suffixes(s, length, idx1, idx2) <= 0:
                    break
                sa[j - gap] = idx2
                sa[j] = idx1
                j -= gap
        gap = gap / 2

    # BWT: last column = char before each sorted suffix
    total = 0
    for i in range(length):
        total += <int>s[(sa[i] - 1 + length) % length]

    free(s)
    free(sa)
    return total


cdef int _compare_suffixes(char *s, int length, int a, int b) nogil:
    """Compare two suffixes of s starting at positions a and b."""
    cdef int k
    for k in range(length):
        if s[(a + k) % length] < s[(b + k) % length]:
            return -1
        elif s[(a + k) % length] > s[(b + k) % length]:
            return 1
    return 0
