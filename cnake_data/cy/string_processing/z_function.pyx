# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""
Z-function computation for a string (Cython-optimized).

Keywords: string processing, z-function, pattern matching, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(500000,))
def z_function(int n):
    """Compute the Z-function and return sum of all Z-values.

    Uses C int arrays for both the string (as char codes) and Z-values.

    Args:
        n: Length of the string.

    Returns:
        Sum of all Z-values.
    """
    cdef int i, l, r
    cdef long long total = 0

    cdef unsigned char *s = <unsigned char *>malloc(n * sizeof(unsigned char))
    if s == NULL:
        raise MemoryError("Failed to allocate string buffer")

    cdef int *z = <int *>malloc(n * sizeof(int))
    if z == NULL:
        free(s)
        raise MemoryError("Failed to allocate Z array")

    # Build deterministic string
    for i in range(n):
        s[i] = 65 + (i * 7 + 3) % 26

    # Initialize Z array
    for i in range(n):
        z[i] = 0

    # Z-algorithm
    l = 0
    r = 0
    for i in range(1, n):
        if i < r:
            z[i] = z[i - l]
            if z[i] > r - i:
                z[i] = r - i
        while i + z[i] < n and s[z[i]] == s[i + z[i]]:
            z[i] += 1
        if i + z[i] > r:
            l = i
            r = i + z[i]

    # Sum all Z-values
    for i in range(n):
        total += z[i]

    free(z)
    free(s)
    return total
