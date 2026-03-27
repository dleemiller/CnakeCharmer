# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""
Numerical integration of f(x)=x^2 from 0 to 1 using the trapezoidal rule (Cython-optimized).

Keywords: integration, trapezoidal, numerical, calculus, cython, benchmark
"""

from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(1000000,))
def trapezoidal_integration(int n):
    """Integrate f(x)=x^2 from 0 to 1 using C-typed trapezoidal rule."""
    cdef double a = 0.0
    cdef double b = 1.0
    cdef double h = (b - a) / n
    cdef double result = 0.5 * (a * a + b * b)
    cdef double x
    cdef int i

    for i in range(1, n):
        x = a + i * h
        result += x * x

    return result * h
