# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Burrows-Wheeler Transform and byte sum of output (Cython-optimized).

Keywords: string processing, burrows-wheeler, bwt, transform, cython, benchmark
"""

from libc.stdlib cimport malloc, free, qsort
from libc.string cimport memcmp
from cnake_charmer.benchmarks import cython_benchmark

# Module-level pointer for qsort comparator
cdef char *_bwt_str = NULL
cdef int _bwt_len = 0


cdef int _cmp_suffix(const void *a, const void *b) noexcept nogil:
    """Compare two suffixes using memcmp where possible."""
    cdef int ia = (<int *>a)[0]
    cdef int ib = (<int *>b)[0]
    cdef int la = _bwt_len - ia
    cdef int lb = _bwt_len - ib
    cdef int min_len = la if la < lb else lb
    cdef int result = memcmp(&_bwt_str[ia], &_bwt_str[ib], min_len)
    if result != 0:
        return result
    return la - lb  # shorter suffix is "smaller"


@cython_benchmark(syntax="cy", args=(5000,))
def burrows_wheeler(int n):
    """Compute BWT of a deterministic string and return sum of output bytes."""
    global _bwt_str, _bwt_len

    cdef int length = n + 1  # +1 for sentinel
    cdef char *s = <char *>malloc(length * sizeof(char))
    cdef int *sa = <int *>malloc(length * sizeof(int))
    if not s or not sa:
        if s: free(s)
        if sa: free(sa)
        raise MemoryError()

    cdef int i, total

    # Generate string with sentinel at end
    for i in range(n):
        s[i] = 65 + (i * 7 + 3) % 4
    s[n] = 0  # sentinel (null byte, sorts first)

    # Initialize suffix array
    for i in range(length):
        sa[i] = i

    # Sort suffixes using qsort + memcmp
    _bwt_str = s
    _bwt_len = length
    qsort(sa, length, sizeof(int), _cmp_suffix)

    # BWT: last column = char before each sorted suffix
    total = 0
    for i in range(length):
        total += <int>s[(sa[i] - 1 + length) % length]

    free(s)
    free(sa)
    return total
