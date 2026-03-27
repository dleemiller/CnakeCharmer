# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Use a cdef class config with readonly attributes to control computation (Cython).

Keywords: algorithms, config, readonly, iteration, tolerance, cdef class, cython, benchmark
"""

from libc.math cimport fabs
from cnake_charmer.benchmarks import cython_benchmark


cdef class Config:
    """Configuration with cdef readonly max_iter and tolerance."""
    cdef readonly int max_iter
    cdef readonly double tolerance

    def __cinit__(self, int max_iter, double tolerance):
        self.max_iter = max_iter
        self.tolerance = tolerance


@cython_benchmark(syntax="cy", args=(100000,))
def config_lookup(int n):
    """Run n independent convergence computations controlled by Config objects."""
    cdef list configs = []
    cdef int i, j, mi
    cdef double tolerance, total, x, prev
    cdef Config cfg

    for i in range(8):
        mi = ((<long long>i * <long long>2654435761 + 17) % 50) + 10
        tolerance = 1.0 / (((<long long>i * <long long>1103515245 + 12345) % 100) + 10)
        configs.append(Config(mi, tolerance))

    total = 0.0
    for i in range(n):
        cfg = <Config>configs[i & 7]
        x = ((<long long>i * <long long>1664525 + <long long>1013904223) % 100000) / 100.0
        for j in range(cfg.max_iter):
            prev = x
            x = x * 0.5 + ((<long long>j * <long long>214013 + i) % 100) / 200.0
            if fabs(x - prev) < cfg.tolerance:
                break
        total += x

    return total
