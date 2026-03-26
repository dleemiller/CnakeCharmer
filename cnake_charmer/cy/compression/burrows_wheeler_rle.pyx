# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Burrows-Wheeler Transform followed by RLE run counting (Cython-optimized).

Keywords: compression, burrows-wheeler, bwt, rle, run length, cython, benchmark
"""

from libc.stdlib cimport malloc, free, qsort
from cnake_charmer.benchmarks import cython_benchmark


cdef char *_bwt_str
cdef int _bwt_len


cdef int _compare_rotations(const void *a, const void *b) noexcept nogil:
    """Compare two rotations of the global string."""
    cdef int ia = (<int *>a)[0]
    cdef int ib = (<int *>b)[0]
    cdef int k
    cdef char ca, cb
    for k in range(_bwt_len):
        ca = _bwt_str[(ia + k) % _bwt_len]
        cb = _bwt_str[(ib + k) % _bwt_len]
        if ca < cb:
            return -1
        elif ca > cb:
            return 1
    return 0


@cython_benchmark(syntax="cy", args=(5000,))
def burrows_wheeler_rle(int n):
    """Compute BWT of a string then count RLE runs."""
    global _bwt_str, _bwt_len

    if n == 0:
        return 0

    cdef char *s = <char *>malloc(n * sizeof(char))
    if not s:
        raise MemoryError()

    cdef int *indices = <int *>malloc(n * sizeof(int))
    if not indices:
        free(s)
        raise MemoryError()

    cdef int i, runs

    # Generate deterministic string
    for i in range(n):
        s[i] = 65 + (i * 7 + 3) % 4

    # Set globals for comparison function
    _bwt_str = s
    _bwt_len = n

    # Initialize indices
    for i in range(n):
        indices[i] = i

    # Sort rotations using qsort
    qsort(indices, n, sizeof(int), _compare_rotations)

    # Extract last column and count runs
    cdef char prev = s[(indices[0] + n - 1) % n]
    cdef char cur
    runs = 1
    for i in range(1, n):
        cur = s[(indices[i] + n - 1) % n]
        if cur != prev:
            runs += 1
        prev = cur

    free(s)
    free(indices)
    return runs
