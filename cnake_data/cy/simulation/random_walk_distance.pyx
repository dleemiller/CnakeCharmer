# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Compute mean squared displacement of deterministic pseudo-random walks (Cython-optimized).

Keywords: random walk, mean squared displacement, simulation, stochastic, cython, benchmark
"""

from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(2000,))
def random_walk_distance(int n):
    """Compute mean squared displacement of n deterministic pseudo-random walks."""
    cdef int steps = 1000
    cdef int w, s, x, y
    cdef long long dx, dy
    cdef double total_sq_dist = 0.0
    cdef unsigned long long h

    for w in range(n):
        x = 0
        y = 0
        h = <unsigned long long>w * 6364136223846793005ULL + 1
        for s in range(steps):
            h = (h * 6364136223846793005ULL + 1442695040888963407ULL)
            dx = (h >> 33) % 3 - 1
            h = (h * 6364136223846793005ULL + 1442695040888963407ULL)
            dy = (h >> 33) % 3 - 1
            x += dx
            y += dy
        total_sq_dist += <double>(x * x + y * y)

    return total_sq_dist / n
