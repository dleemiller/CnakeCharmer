# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Count palindromic substrings and find longest using Manacher's algorithm (Cython).

Keywords: string processing, palindrome, manacher, algorithm, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(500000,))
def manacher(int n):
    """Find longest palindrome and count all palindromic substrings."""
    cdef int i, center, right, mirror, a, b
    cdef int t_len
    cdef int *t
    cdef int *p
    cdef long long total
    cdef int max_len, max_center_t, center_pos
    cdef unsigned int seed

    if n <= 0:
        return (0, 0, 0)

    t_len = 2 * n + 1
    t = <int *>malloc(t_len * sizeof(int))
    p = <int *>malloc(t_len * sizeof(int))
    if not t or not p:
        free(t); free(p)
        raise MemoryError()

    # Build transformed string using xorshift PRNG
    for i in range(t_len):
        t[i] = 99  # sentinel
    seed = 42
    for i in range(n):
        seed ^= (seed << 13) & 0xFFFFFFFF
        seed ^= (seed >> 17) & 0xFFFFFFFF
        seed ^= (seed << 5) & 0xFFFFFFFF
        t[2 * i + 1] = seed % 3

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

    max_len = 0
    max_center_t = 0
    total = 0
    for i in range(t_len):
        total += (p[i] + 1) / 2
        if p[i] > max_len:
            max_len = p[i]
            max_center_t = i

    center_pos = max_center_t / 2

    free(t)
    free(p)
    return (max_len, center_pos, total)
