# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Wildcard pattern matching against generated strings (Cython-optimized).

Keywords: dynamic programming, wildcard, pattern matching, string, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from libc.string cimport memset
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(200000,))
def wildcard_matching(int n):
    """Count how many generated strings match pattern 'D*?D' using DP."""
    cdef int count = 0
    cdef int i, j, pi, si
    cdef int pat_len = 4  # "D*?D"
    cdef int s_len = 5
    cdef char pattern[5]
    cdef char s[6]
    cdef int *prev
    cdef int *curr
    cdef char pc

    pattern[0] = ord('D')
    pattern[1] = ord('*')
    pattern[2] = ord('?')
    pattern[3] = ord('D')
    pattern[4] = 0

    prev = <int *>malloc((s_len + 1) * sizeof(int))
    curr = <int *>malloc((s_len + 1) * sizeof(int))
    if not prev or not curr:
        if prev: free(prev)
        if curr: free(curr)
        raise MemoryError()

    for i in range(n):
        # Generate string
        for j in range(5):
            s[j] = 65 + (i * j + 3) % 26
        s[5] = 0

        # DP: prev[si] = can pattern[pi+1:] match s[si:]
        memset(prev, 0, (s_len + 1) * sizeof(int))
        prev[s_len] = 1  # empty pattern matches empty string

        for pi in range(pat_len - 1, -1, -1):
            memset(curr, 0, (s_len + 1) * sizeof(int))
            pc = pattern[pi]
            if pc == ord('*'):
                curr[s_len] = prev[s_len]
                for si in range(s_len - 1, -1, -1):
                    curr[si] = prev[si] or curr[si + 1]
            else:
                for si in range(s_len - 1, -1, -1):
                    if pc == ord('?') or pc == s[si]:
                        curr[si] = prev[si + 1]
            # Swap prev and curr
            prev, curr = curr, prev

        if prev[0]:
            count += 1

    free(prev)
    free(curr)
    return count
