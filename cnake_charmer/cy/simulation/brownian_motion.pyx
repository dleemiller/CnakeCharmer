# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Simulate Brownian motion for n particles (Cython-optimized).

Keywords: brownian motion, simulation, random walk, particle, displacement, cython, benchmark
"""

from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(50000,))
def brownian_motion(int n):
    """Simulate n particles doing Brownian motion for 100 steps."""
    cdef int steps = 100
    cdef double total_sq_disp = 0.0
    cdef double x, y, dx, dy
    cdef unsigned long long h, h2
    cdef int p, s

    for p in range(n):
        x = 0.0
        y = 0.0
        for s in range(steps):
            # Deterministic LCG-style hash
            h = ((<unsigned long long>s * 6364136223846793005ULL +
                  <unsigned long long>p * 1442695040888963407ULL + 1) >> 16) & 0xFFFFFFFF
            dx = (<double>(h % 201) - 100.0) / 100.0
            h2 = ((h * 6364136223846793005ULL + 1) >> 16) & 0xFFFFFFFF
            dy = (<double>(h2 % 201) - 100.0) / 100.0
            x += dx
            y += dy
        total_sq_disp += x * x + y * y

    return total_sq_disp / <double>n
