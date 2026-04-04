# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Find longest common substring of two deterministic strings (Cython).

Keywords: string processing, longest common substring, dynamic programming, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from libc.string cimport memset
from cnake_data.benchmarks import cython_benchmark


cdef void _lcs_kernel(
    int n,
    int* sa,
    int* sb,
    int* max_len_out,
    int* start_a_out,
    int* start_b_out,
) noexcept nogil:
    cdef int *prev = <int *>malloc((n + 1) * sizeof(int))
    cdef int *curr = <int *>malloc((n + 1) * sizeof(int))
    cdef int *temp
    cdef int i, j
    cdef int max_len = 0
    cdef int start_a = 0
    cdef int start_b = 0

    if not prev or not curr:
        if prev:
            free(prev)
        if curr:
            free(curr)
        max_len_out[0] = 0
        start_a_out[0] = 0
        start_b_out[0] = 0
        return

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
        temp = prev
        prev = curr
        curr = temp
        memset(curr, 0, (n + 1) * sizeof(int))

    free(prev)
    free(curr)
    max_len_out[0] = max_len
    start_a_out[0] = start_a
    start_b_out[0] = start_b


@cython_benchmark(syntax="cy", args=(2000,))
def longest_common_substring_rolling(int n):
    """Find longest common substring using DP with rolling row."""
    cdef int *sa = <int *>malloc(n * sizeof(int))
    cdef int *sb = <int *>malloc(n * sizeof(int))
    if not sa or not sb:
        free(sa); free(sb)
        raise MemoryError()

    cdef int i
    cdef int max_len = 0
    cdef int start_a = 0
    cdef int start_b = 0
    cdef unsigned int s, mask = 0xFFFFFFFF

    s = 12345
    for i in range(n):
        s ^= (s << 13) & mask
        s ^= (s >> 17) & mask
        s ^= (s << 5) & mask
        sa[i] = s % 4

    s = 67890
    for i in range(n):
        s ^= (s << 13) & mask
        s ^= (s >> 17) & mask
        s ^= (s << 5) & mask
        sb[i] = s % 4

    with nogil:
        _lcs_kernel(n, sa, sb, &max_len, &start_a, &start_b)

    free(sa)
    free(sb)
    return (max_len, start_a, start_b)
