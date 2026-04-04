# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Find longest common substring of two deterministic strings (Cython).

Keywords: string processing, longest common substring, dynamic programming, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from libc.string cimport memset
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(5000,))
def longest_common_substring(int n):
    """Find longest common substring using DP with rolling row."""
    cdef int *sa = <int *>malloc(n * sizeof(int))
    cdef int *sb = <int *>malloc(n * sizeof(int))
    cdef int *prev = <int *>malloc((n + 1) * sizeof(int))
    cdef int *curr = <int *>malloc((n + 1) * sizeof(int))
    cdef int *temp
    if not sa or not sb or not prev or not curr:
        free(sa); free(sb); free(prev); free(curr)
        raise MemoryError()

    cdef int i, j
    cdef int max_len = 0
    cdef int start_a = 0
    cdef int start_b = 0
    cdef unsigned int s

    # Generate string A with xorshift PRNG
    s = 12345
    for i in range(n):
        s ^= (s << 13) & 0xFFFFFFFF
        s ^= (s >> 17) & 0xFFFFFFFF
        s ^= (s << 5) & 0xFFFFFFFF
        sa[i] = s % 4

    # Generate string B with xorshift PRNG
    s = 67890
    for i in range(n):
        s ^= (s << 13) & 0xFFFFFFFF
        s ^= (s >> 17) & 0xFFFFFFFF
        s ^= (s << 5) & 0xFFFFFFFF
        sb[i] = s % 4

    memset(prev, 0, (n + 1) * sizeof(int))
    memset(curr, 0, (n + 1) * sizeof(int))

    for i in range(1, n + 1):
        for j in range(1, n + 1):
            if sa[i - 1] == sb[j - 1]:
                curr[j] = prev[j - 1] + 1
                if curr[j] > max_len:
                    max_len = curr[j]
                    start_a = i - max_len
                    start_b = j - max_len
            else:
                curr[j] = 0
        # Swap rows
        temp = prev
        prev = curr
        curr = temp
        memset(curr, 0, (n + 1) * sizeof(int))

    free(sa)
    free(sb)
    free(prev)
    free(curr)
    return (max_len, start_a, start_b)
