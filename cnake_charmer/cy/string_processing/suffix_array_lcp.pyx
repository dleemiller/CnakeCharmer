# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Build suffix array + LCP array, return sum of LCP values (Cython-optimized).

Keywords: string processing, suffix array, lcp, longest common prefix, cython, benchmark
"""

from libc.stdlib cimport malloc, free, qsort
from cnake_charmer.benchmarks import cython_benchmark


# Global pointer for qsort comparison
cdef char *_sort_str
cdef int _sort_len


cdef int suffix_compare(const void *a, const void *b) noexcept nogil:
    """Compare two suffixes for qsort."""
    cdef int ia = (<int *>a)[0]
    cdef int ib = (<int *>b)[0]
    cdef int i = 0
    cdef int la = _sort_len - ia
    cdef int lb = _sort_len - ib
    cdef int min_len = la if la < lb else lb

    while i < min_len:
        if _sort_str[ia + i] != _sort_str[ib + i]:
            return <int>_sort_str[ia + i] - <int>_sort_str[ib + i]
        i += 1

    return la - lb


@cython_benchmark(syntax="cy", args=(50000,))
def suffix_array_lcp(int n):
    """Build suffix array + LCP using C arrays and qsort."""
    global _sort_str, _sort_len

    cdef int i, j, k
    cdef char *s
    cdef int *sa
    cdef int *rank_arr
    cdef int *lcp
    cdef long long total

    if n <= 0:
        return 0

    s = <char *>malloc(n * sizeof(char))
    sa = <int *>malloc(n * sizeof(int))
    rank_arr = <int *>malloc(n * sizeof(int))
    lcp = <int *>malloc(n * sizeof(int))
    if not s or not sa or not rank_arr or not lcp:
        raise MemoryError()

    # Build string
    for i in range(n):
        s[i] = 65 + (i * 7 + 3) % 4

    # Build suffix array
    for i in range(n):
        sa[i] = i

    _sort_str = s
    _sort_len = n
    qsort(sa, n, sizeof(int), suffix_compare)

    # Build rank array
    for i in range(n):
        rank_arr[sa[i]] = i

    # Kasai's LCP algorithm
    k = 0
    for i in range(n):
        if rank_arr[i] == 0:
            k = 0
            lcp[0] = 0
            continue
        j = sa[rank_arr[i] - 1]
        while i + k < n and j + k < n and s[i + k] == s[j + k]:
            k += 1
        lcp[rank_arr[i]] = k
        if k > 0:
            k -= 1

    total = 0
    for i in range(n):
        total += lcp[i]

    free(s)
    free(sa)
    free(rank_arr)
    free(lcp)
    return total
