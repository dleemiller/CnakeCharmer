# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Estimate pi using deterministic quasi-random points (Cython-optimized).

Keywords: simulation, monte carlo, pi, quasi-random, halton, estimation, cython, benchmark
"""

from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(500000,))
def monte_carlo_pi(int n):
    """Estimate pi using n deterministic quasi-random points via Halton sequences."""
    cdef int points_inside = 0
    cdef double last_dist = 0.0
    cdef double x, y, f, dist
    cdef int i, idx

    for i in range(1, n + 1):
        # Halton base 2
        x = 0.0
        f = 0.5
        idx = i
        while idx > 0:
            x += f * (idx % 2)
            idx = idx // 2
            f *= 0.5

        # Halton base 3
        y = 0.0
        f = 1.0 / 3.0
        idx = i
        while idx > 0:
            y += f * (idx % 3)
            idx = idx // 3
            f /= 3.0

        dist = x * x + y * y
        last_dist = dist
        if dist <= 1.0:
            points_inside += 1

    cdef double estimate = 4.0 * points_inside / n

    return (estimate, points_inside, last_dist)
