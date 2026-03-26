# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Find length of longest repeated substring (Cython-optimized).

Keywords: suffix array, lcp, longest repeated substring, string, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(10000,))
def longest_repeated_substring(int n):
    """Find longest repeated substring using suffix array + LCP with C arrays."""
    cdef char *s = <char *>malloc(n * sizeof(char))
    if not s:
        raise MemoryError()

    cdef int i, a, b, lcp_len, best

    # Generate string
    for i in range(n):
        s[i] = <char>(65 + (i * 7 + 3) % 26)

    # Build suffix array using Python sort (suffix array construction in pure C
    # would be complex; we convert to Python for the sort step)
    cdef bytes py_s = s[:n]
    sa = list(range(n))
    sa.sort(key=lambda idx: py_s[idx:])

    # Compute LCP with C-level character comparison
    best = 0
    for i in range(1, n):
        a = sa[i - 1]
        b = sa[i]
        lcp_len = 0
        while a + lcp_len < n and b + lcp_len < n and s[a + lcp_len] == s[b + lcp_len]:
            lcp_len += 1
        if lcp_len > best:
            best = lcp_len

    free(s)
    return best
