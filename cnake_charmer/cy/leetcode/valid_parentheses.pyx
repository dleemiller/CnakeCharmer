# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Count valid parentheses strings among deterministically generated strings.

Keywords: leetcode, valid parentheses, stack, string, cython, benchmark
"""

from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(500000,))
def valid_parentheses(int n):
    """Count how many of n generated bracket sequences are valid."""
    cdef int count = 0
    cdef int i, j, depth
    cdef unsigned int bits
    cdef bint valid

    for i in range(n):
        bits = (((<unsigned int>i) * 2654435761U) >> 4) & 0xFF
        depth = 0
        valid = True
        for j in range(8):
            if (bits >> j) & 1 == 0:
                depth += 1
            else:
                depth -= 1
            if depth < 0:
                valid = False
                break
        if valid and depth == 0:
            count += 1

    return count
