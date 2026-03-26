# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Minimum cuts to partition a deterministic string into palindromes (Cython-optimized).

Keywords: palindrome, partition, dynamic programming, string, minimum cuts, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(2000,))
def palindrome_partition(int n):
    """Compute minimum cuts to partition s into palindromes."""
    cdef char *s = <char *>malloc(n * sizeof(char))
    cdef unsigned char *is_pal = <unsigned char *>malloc(n * n * sizeof(unsigned char))
    cdef int *cuts = <int *>malloc(n * sizeof(int))

    if not s or not is_pal or not cuts:
        if s:
            free(s)
        if is_pal:
            free(is_pal)
        if cuts:
            free(cuts)
        raise MemoryError()

    cdef int i, j, length
    cdef int result

    # Generate string: s[i] = 65 + (i*i + 3*i + 1) % 4
    for i in range(n):
        s[i] = 65 + (i * i + 3 * i + 1) % 4

    # Initialize is_pal to 0
    for i in range(n * n):
        is_pal[i] = 0

    # Single chars are palindromes
    for i in range(n):
        is_pal[i * n + i] = 1

    # Length 2
    for i in range(n - 1):
        if s[i] == s[i + 1]:
            is_pal[i * n + i + 1] = 1

    # Length 3+
    for length in range(3, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            if s[i] == s[j] and is_pal[(i + 1) * n + (j - 1)]:
                is_pal[i * n + j] = 1

    # DP for minimum cuts
    for i in range(n):
        cuts[i] = i  # worst case
    for i in range(1, n):
        if is_pal[0 * n + i]:
            cuts[i] = 0
            continue
        for j in range(1, i + 1):
            if is_pal[j * n + i]:
                if cuts[j - 1] + 1 < cuts[i]:
                    cuts[i] = cuts[j - 1] + 1

    result = cuts[n - 1]

    free(s)
    free(is_pal)
    free(cuts)

    return result
