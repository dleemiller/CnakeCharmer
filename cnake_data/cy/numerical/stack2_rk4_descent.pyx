# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Trace a 2D RK4 descent path in a linear gradient field (Cython).

Adapted from The Stack v2 Cython candidate:
- blob_id: 731b31ede4e99a6989cdd0d6add9b508e2c754e1
- filename: optimal_path_cython.pyx
"""

from libc.math cimport sqrt
from cnake_data.benchmarks import cython_benchmark


cdef inline double _grad_x(double coord_x, double coord_y) noexcept nogil:
    return 2.0 * coord_x + 0.25 * coord_y


cdef inline double _grad_y(double coord_x, double coord_y) noexcept nogil:
    return 2.0 * coord_y - 0.125 * coord_x


cdef void _rk4_kernel(
    int start_x_milli,
    int start_y_milli,
    int step_count,
    int step_milli,
    int min_steps,
    int* out_x,
    int* out_y,
    unsigned int* out_travel,
    int* out_iter,
) noexcept nogil:
    cdef double coord_x = start_x_milli / 1000.0
    cdef double coord_y = start_y_milli / 1000.0
    cdef double delta = step_milli / 1000.0
    cdef double half_delta = 0.5 * delta
    cdef double step_scale = delta / 6.0
    cdef double tol = 1e-18
    cdef double k1x, k1y, k2x, k2y, k3x, k3y, k4x, k4y
    cdef double next_x, next_y, dx, dy, dist2
    cdef double travel = 0.0
    cdef int idx
    cdef int stop_iter = step_count

    for idx in range(step_count):
        k1x = -_grad_x(coord_x, coord_y)
        k1y = -_grad_y(coord_x, coord_y)
        k2x = -_grad_x(coord_x + half_delta * k1x, coord_y + half_delta * k1y)
        k2y = -_grad_y(coord_x + half_delta * k1x, coord_y + half_delta * k1y)
        k3x = -_grad_x(coord_x + half_delta * k2x, coord_y + half_delta * k2y)
        k3y = -_grad_y(coord_x + half_delta * k2x, coord_y + half_delta * k2y)
        k4x = -_grad_x(coord_x + delta * k3x, coord_y + delta * k3y)
        k4y = -_grad_y(coord_x + delta * k3x, coord_y + delta * k3y)

        next_x = coord_x + step_scale * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
        next_y = coord_y + step_scale * (k1y + 2.0 * k2y + 2.0 * k3y + k4y)

        dx = next_x - coord_x
        dy = next_y - coord_y
        dist2 = dx * dx + dy * dy
        travel += sqrt(dist2)

        coord_x = next_x
        coord_y = next_y

        if (idx + 1) >= min_steps and dist2 < tol:
            stop_iter = idx + 1
            break

    out_x[0] = <int>(coord_x * 1000000.0)
    out_y[0] = <int>(coord_y * 1000000.0)
    out_travel[0] = <unsigned int>(travel * 1000000.0)
    out_iter[0] = stop_iter


@cython_benchmark(syntax="cy", args=(1250, -980, 220000, 17, 200000))
def stack2_rk4_descent(
    int start_x_milli, int start_y_milli, int step_count, int step_milli, int min_steps=0
):
    cdef int out_x = 0
    cdef int out_y = 0
    cdef unsigned int out_travel = 0
    cdef int out_iter = step_count
    with nogil:
        _rk4_kernel(
            start_x_milli,
            start_y_milli,
            step_count,
            step_milli,
            min_steps,
            &out_x,
            &out_y,
            &out_travel,
            &out_iter,
        )
    return (out_x, out_y, out_travel, out_iter)
