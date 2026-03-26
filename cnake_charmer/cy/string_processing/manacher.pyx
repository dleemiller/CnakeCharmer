# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Count total palindromic substrings using Manacher's algorithm (Cython-optimized).

Keywords: string processing, palindrome, manacher, algorithm, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(500000,))
def manacher(int n):
    """Count palindromic substrings using Manacher's with C arrays."""
    cdef int i, center, right, mirror, a, b
    cdef int t_len
    cdef char *t
    cdef int *p
    cdef long long total

    if n <= 0:
        return 0

    t_len = 2 * n + 1
    t = <char *>malloc(t_len * sizeof(char))
    p = <int *>malloc(t_len * sizeof(int))
    if not t or not p:
        raise MemoryError()

    # Build transformed string
    for i in range(t_len):
        t[i] = 35  # '#'
    for i in range(n):
        t[2 * i + 1] = 65 + (i * 7 + 3) % 26

    # Manacher's
    center = 0
    right = 0
    for i in range(t_len):
        mirror = 2 * center - i
        if i < right:
            p[i] = right - i
            if mirror >= 0 and p[mirror] < p[i]:
                p[i] = p[mirror]
        else:
            p[i] = 0

        a = i + p[i] + 1
        b = i - p[i] - 1
        while a < t_len and b >= 0 and t[a] == t[b]:
            p[i] += 1
            a += 1
            b -= 1

        if i + p[i] > right:
            center = i
            right = i + p[i]

    total = 0
    for i in range(t_len):
        total += (p[i] + 1) / 2

    free(t)
    free(p)
    return total
