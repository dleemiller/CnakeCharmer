# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Deterministic simulated annealing on Rastrigin function (Cython-optimized).

Keywords: optimization, simulated annealing, Rastrigin, deterministic, cython, benchmark
"""

from libc.math cimport cos, sin, exp, M_PI
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(2000000,))
def simulated_annealing_deterministic(int n):
    """Deterministic simulated annealing to minimize 2D Rastrigin function."""
    cdef int i
    cdef double A = 10.0
    cdef double x = 2.5, y = 2.5
    cdef double x_new, y_new, dx, dy
    cdef double f_curr, f_new, delta, temp, t_frac
    cdef double best_x, best_y, best_f
    cdef double threshold, boltz
    cdef double two_pi = 2.0 * M_PI
    cdef int accept

    # Rastrigin at initial point
    f_curr = 2.0 * A + (x * x - A * cos(two_pi * x)) + (y * y - A * cos(two_pi * y))
    best_x = x
    best_y = y
    best_f = f_curr

    for i in range(n):
        t_frac = <double>i / <double>n
        temp = 10.0 * (1.0 - t_frac)

        dx = 0.1 * (1.0 - t_frac) * cos(i * 0.1)
        dy = 0.1 * (1.0 - t_frac) * sin(i * 0.13)

        x_new = x + dx
        y_new = y + dy

        f_new = 2.0 * A + (x_new * x_new - A * cos(two_pi * x_new)) + \
                (y_new * y_new - A * cos(two_pi * y_new))
        delta = f_new - f_curr

        accept = 0
        if delta < 0.0:
            accept = 1
        elif temp > 1e-10:
            threshold = 0.5 + 0.3 * sin(i * 0.07)
            if delta / temp < 50.0:
                boltz = exp(-delta / temp)
            else:
                boltz = 0.0
            if boltz > threshold:
                accept = 1

        if accept:
            x = x_new
            y = y_new
            f_curr = f_new
            if f_curr < best_f:
                best_f = f_curr
                best_x = x
                best_y = y

    return (best_x, best_y, best_f)
