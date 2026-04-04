# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""SIR epidemic model simulation (Cython-optimized).

Keywords: SIR model, epidemic, simulation, population dynamics, cython, benchmark
"""

from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(10000000,))
def epidemic_sir(int n):
    """Simulate SIR epidemic model."""
    cdef double s = <double>(n - 10)
    cdef double inf = 10.0
    cdef double r = 0.0
    cdef double beta = 0.3 / n
    cdef double gamma_val = 0.1
    cdef double dt = 0.1
    cdef double new_infections, new_recoveries
    cdef int t

    for t in range(1000):
        new_infections = beta * s * inf * dt
        new_recoveries = gamma_val * inf * dt
        s -= new_infections
        inf += new_infections - new_recoveries
        r += new_recoveries

    return int(r)
