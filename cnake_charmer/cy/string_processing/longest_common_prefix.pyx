# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""
Total longest common prefix length between consecutive string pairs (Cython-optimized).

Keywords: string processing, longest common prefix, comparison, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(50000,))
def longest_common_prefix(int n):
    """Find total LCP length using char* comparison in C."""
    cdef int i, j, k, lcp
    cdef long long total = 0
    cdef int max_lcp = 0
    cdef int str_len = 20
    cdef unsigned char *data = <unsigned char *>malloc(n * str_len * sizeof(unsigned char))
    cdef unsigned char *s1
    cdef unsigned char *s2

    if data == NULL:
        raise MemoryError("Failed to allocate array")

    # Generate all strings as flat byte array
    for i in range(n):
        for j in range(str_len):
            data[i * str_len + j] = 65 + (i * j + 3) % 26

    # Compare consecutive pairs
    for i in range(n - 1):
        s1 = &data[i * str_len]
        s2 = &data[(i + 1) * str_len]
        lcp = 0
        for k in range(str_len):
            if s1[k] == s2[k]:
                lcp += 1
            else:
                break
        total += lcp
        if lcp > max_lcp:
            max_lcp = lcp

    cdef long long result_total = total
    cdef int result_max = max_lcp
    free(data)
    return (result_total, result_max)
