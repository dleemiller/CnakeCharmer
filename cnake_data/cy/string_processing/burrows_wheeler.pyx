# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Burrows-Wheeler Transform returning character codes and primary index.

Keywords: string processing, burrows-wheeler, bwt, transform, cython, benchmark
"""

from libc.stdlib cimport malloc, free, qsort
from libc.string cimport memcmp
from cnake_data.benchmarks import cython_benchmark

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
    return la - lb


@cython_benchmark(syntax="cy", args=(50000,))
def burrows_wheeler(int n):
    """Compute BWT of a deterministic string and return summary."""
    global _bwt_str, _bwt_len

    cdef int length = n + 1  # +1 for sentinel
    cdef char *s = <char *>malloc(length * sizeof(char))
    cdef int *sa = <int *>malloc(length * sizeof(int))
    if not s or not sa:
        if s: free(s)
        if sa: free(sa)
        raise MemoryError()

    cdef int i, bwt_first, bwt_last, primary_index

    # Generate string with sentinel at end
    for i in range(n):
        s[i] = 65 + (i * 7 + 3) % 4
    s[n] = 0  # sentinel (null byte)

    # Initialize suffix array
    for i in range(length):
        sa[i] = i

    # Sort suffixes
    _bwt_str = s
    _bwt_len = length
    qsort(sa, length, sizeof(int), _cmp_suffix)

    # BWT first and last char
    bwt_first = <int>s[(sa[0] - 1 + length) % length]
    bwt_last = <int>s[(sa[length - 1] - 1 + length) % length]

    # Primary index: where rotation 0 appears
    primary_index = 0
    for i in range(length):
        if sa[i] == 0:
            primary_index = i
            break

    free(s)
    free(sa)
    return (bwt_first, bwt_last, primary_index)
