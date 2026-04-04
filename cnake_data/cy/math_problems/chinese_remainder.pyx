# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""
Chinese Remainder Theorem — solve and sum congruence systems (Cython-optimized).

Keywords: math, chinese remainder theorem, CRT, extended GCD, number theory, cython, benchmark
"""

from cnake_data.benchmarks import cython_benchmark


cdef inline void ext_gcd(long long a, long long b, long long *g, long long *x, long long *y):
    """Extended GCD: compute g, x, y such that a*x + b*y = g."""
    cdef long long g1, x1, y1
    if b == 0:
        g[0] = a
        x[0] = 1
        y[0] = 0
        return
    ext_gcd(b, a % b, &g1, &x1, &y1)
    g[0] = g1
    x[0] = y1
    y[0] = x1 - (a / b) * y1


@cython_benchmark(syntax="cy", args=(100000,))
def chinese_remainder(int n):
    """Solve pairwise CRT systems using typed extended GCD."""
    cdef long long MOD = 1000000007
    cdef long long total = 0
    cdef long long a1, m1, a2, m2, g, p, q, lcm, diff, solution, k
    cdef int i

    for i in range(n):
        a1 = (i * 7 + 3) % 100
        m1 = i * 3 + 11
        a2 = ((i + 1) * 7 + 3) % 100
        m2 = (i + 1) * 3 + 11

        ext_gcd(m1, m2, &g, &p, &q)
        lcm = m1 * (m2 / g)
        diff = a2 - a1
        if diff % g != 0:
            continue
        k = (diff / g) % (m2 / g)
        solution = (a1 + m1 * k * p) % lcm
        if solution < 0:
            solution += lcm
        total = (total + solution) % MOD

    return int(total)
