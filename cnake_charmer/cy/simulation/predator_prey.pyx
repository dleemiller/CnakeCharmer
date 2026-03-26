# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Lotka-Volterra predator-prey simulation (Cython-optimized).

Keywords: simulation, predator, prey, Lotka-Volterra, ODE, cython, benchmark
"""

from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(5000,))
def predator_prey(int n):
    """Simulate Lotka-Volterra predator-prey dynamics for n*100 steps.

    Args:
        n: Scale factor; total steps = n * 100.

    Returns:
        Final prey population as a float.
    """
    cdef double dt = 0.001
    cdef int steps = n * 100
    cdef double prey = 100.0
    cdef double predator = 20.0
    cdef double alpha = 0.1
    cdef double beta = 0.002
    cdef double gamma = 0.4
    cdef double delta = 0.001
    cdef double dprey, dpredator
    cdef int i

    for i in range(steps):
        dprey = (alpha * prey - beta * prey * predator) * dt
        dpredator = (delta * prey * predator - gamma * predator) * dt
        prey += dprey
        predator += dpredator

    return prey
