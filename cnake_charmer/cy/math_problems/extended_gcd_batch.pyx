# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Compute extended GCD for n deterministic pairs (Cython-optimized).

Keywords: extended euclidean, gcd, number theory, bezout, batch, cython, benchmark
"""

from cnake_charmer.benchmarks import cython_benchmark


cdef long long _extended_gcd_x(long long a, long long b):
    """Return x such that a*x + b*y = gcd(a, b). Iterative."""
    cdef long long old_r, r, old_s, s, q, tmp
    old_r = a
    r = b
    old_s = 1
    s = 0
    while r != 0:
        q = old_r / r
        tmp = old_r - q * r
        old_r = r
        r = tmp
        tmp = old_s - q * s
        old_s = s
        s = tmp
    return old_s


@cython_benchmark(syntax="cy", args=(1000000,))
def extended_gcd_batch(int n):
    """Compute extended GCD for n pairs and sum absolute x coefficients."""
    cdef long long total, a, b, x
    cdef int i

    total = 0
    for i in range(n):
        a = (i * 7 + 3) % 1000 + 1
        b = (i * 13 + 7) % 1000 + 1
        x = _extended_gcd_x(a, b)
        if x < 0:
            x = -x
        total += x

    return int(total)
