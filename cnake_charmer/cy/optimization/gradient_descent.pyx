# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Minimize Rosenbrock function using gradient descent (Cython-optimized).

Keywords: gradient descent, rosenbrock, optimization, minimization, cython, benchmark
"""

from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(1000000,))
def gradient_descent(int n):
    """Minimize Rosenbrock f(x,y) via gradient descent."""
    cdef double x = -1.0
    cdef double y = 1.0
    cdef double lr = 0.001
    cdef double dx, dy, f_val
    cdef int i

    for i in range(n):
        dx = -400.0 * x * (y - x * x) - 2.0 * (1.0 - x)
        dy = 200.0 * (y - x * x)
        x -= lr * dx
        y -= lr * dy

    f_val = 100.0 * (y - x * x) * (y - x * x) + (1.0 - x) * (1.0 - x)
    return f_val
