# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Newton-Raphson iteration on a cubic polynomial (Cython).

Sourced from SFT DuckDB blob: 055f6686df5aedcd5115738a7d744e3de7d821e7
Keywords: newton raphson, polynomial root, numerical method, iteration, cython
"""

from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(1.75, 2000000, 1.0))
def polynomial_newton_steps(double start, int iterations, double cubic_scale):
    cdef int i
    cdef double x = start
    cdef double fx, dfx, x_next
    cdef double step_sum = 0.0
    cdef double step, avg_step

    for i in range(iterations):
        fx = cubic_scale * x * x * x - 2.0 * x - 1.0
        dfx = 3.0 * cubic_scale * x * x - 2.0
        if dfx == 0.0:
            break
        x_next = x - fx / dfx
        step = x_next - x
        if step < 0.0:
            step = -step
        step_sum += step
        x = x_next

    fx = cubic_scale * x * x * x - 2.0 * x - 1.0
    avg_step = step_sum / iterations if iterations > 0 else 0.0
    return (round(x, 12), round(fx, 12), round(avg_step, 12))
