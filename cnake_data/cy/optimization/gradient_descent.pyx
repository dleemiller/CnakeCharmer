# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Gradient descent on Rosenbrock function from n starting points (Cython-optimized).

Runs gradient descent from multiple deterministic starting points and
returns the best result found.

Keywords: gradient descent, rosenbrock, optimization, minimization, cython, benchmark
"""

from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(2000,))
def gradient_descent(int n):
    """Minimize Rosenbrock f(x,y) via gradient descent from n starting points."""
    cdef double best_x = 0.0, best_y = 0.0, best_val = 1e30
    cdef double lr = 0.001
    cdef int iters = 500
    cdef int s, j
    cdef double x, y, dx, dy, f_val

    for s in range(n):
        # Deterministic starting point
        x = -2.0 + 4.0 * ((s * 7 + 3) % n) / n
        y = -2.0 + 4.0 * ((s * 13 + 7) % n) / n

        for j in range(iters):
            dx = -400.0 * x * (y - x * x) - 2.0 * (1.0 - x)
            dy = 200.0 * (y - x * x)
            x -= lr * dx
            y -= lr * dy

        f_val = 100.0 * (y - x * x) * (y - x * x) + (1.0 - x) * (1.0 - x)
        if f_val < best_val:
            best_val = f_val
            best_x = x
            best_y = y

    return (best_x, best_y, best_val)
