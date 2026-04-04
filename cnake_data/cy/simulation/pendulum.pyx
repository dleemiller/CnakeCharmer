# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Double pendulum simulation using RK4 integration (Cython-optimized).

Keywords: simulation, double pendulum, RK4, physics, ODE, cython, benchmark
"""

from libc.math cimport sin, cos, M_PI
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(200000,))
def pendulum(int n):
    """Simulate double pendulum using RK4 with typed variables."""
    cdef double dt = 0.001
    cdef double g = 9.81
    cdef double m1 = 1.0
    cdef double m2 = 1.0
    cdef double L1 = 1.0
    cdef double L2 = 1.0
    cdef int i

    cdef double th1 = M_PI / 4.0
    cdef double th2 = M_PI / 2.0
    cdef double w1 = 0.0
    cdef double w2 = 0.0

    cdef double delta, sin_d, cos_d, den1, den2

    cdef double k1_th1, k1_w1, k1_th2, k1_w2
    cdef double k2_th1, k2_w1, k2_th2, k2_w2
    cdef double k3_th1, k3_w1, k3_th2, k3_w2
    cdef double k4_th1, k4_w1, k4_th2, k4_w2
    cdef double a_th1, a_w1, a_th2, a_w2

    for i in range(n):
        # k1
        delta = th2 - th1
        sin_d = sin(delta)
        cos_d = cos(delta)
        den1 = (m1 + m2) * L1 - m2 * L1 * cos_d * cos_d
        den2 = (L2 / L1) * den1
        k1_th1 = w1
        k1_w1 = (m2 * L1 * w1 * w1 * sin_d * cos_d
                  + m2 * g * sin(th2) * cos_d
                  + m2 * L2 * w2 * w2 * sin_d
                  - (m1 + m2) * g * sin(th1)) / den1
        k1_th2 = w2
        k1_w2 = (-m2 * L2 * w2 * w2 * sin_d * cos_d
                  + (m1 + m2) * g * sin(th1) * cos_d
                  - (m1 + m2) * L1 * w1 * w1 * sin_d
                  - (m1 + m2) * g * sin(th2)) / den2

        # k2
        a_th1 = th1 + 0.5 * dt * k1_th1
        a_w1 = w1 + 0.5 * dt * k1_w1
        a_th2 = th2 + 0.5 * dt * k1_th2
        a_w2 = w2 + 0.5 * dt * k1_w2
        delta = a_th2 - a_th1
        sin_d = sin(delta)
        cos_d = cos(delta)
        den1 = (m1 + m2) * L1 - m2 * L1 * cos_d * cos_d
        den2 = (L2 / L1) * den1
        k2_th1 = a_w1
        k2_w1 = (m2 * L1 * a_w1 * a_w1 * sin_d * cos_d
                  + m2 * g * sin(a_th2) * cos_d
                  + m2 * L2 * a_w2 * a_w2 * sin_d
                  - (m1 + m2) * g * sin(a_th1)) / den1
        k2_th2 = a_w2
        k2_w2 = (-m2 * L2 * a_w2 * a_w2 * sin_d * cos_d
                  + (m1 + m2) * g * sin(a_th1) * cos_d
                  - (m1 + m2) * L1 * a_w1 * a_w1 * sin_d
                  - (m1 + m2) * g * sin(a_th2)) / den2

        # k3
        a_th1 = th1 + 0.5 * dt * k2_th1
        a_w1 = w1 + 0.5 * dt * k2_w1
        a_th2 = th2 + 0.5 * dt * k2_th2
        a_w2 = w2 + 0.5 * dt * k2_w2
        delta = a_th2 - a_th1
        sin_d = sin(delta)
        cos_d = cos(delta)
        den1 = (m1 + m2) * L1 - m2 * L1 * cos_d * cos_d
        den2 = (L2 / L1) * den1
        k3_th1 = a_w1
        k3_w1 = (m2 * L1 * a_w1 * a_w1 * sin_d * cos_d
                  + m2 * g * sin(a_th2) * cos_d
                  + m2 * L2 * a_w2 * a_w2 * sin_d
                  - (m1 + m2) * g * sin(a_th1)) / den1
        k3_th2 = a_w2
        k3_w2 = (-m2 * L2 * a_w2 * a_w2 * sin_d * cos_d
                  + (m1 + m2) * g * sin(a_th1) * cos_d
                  - (m1 + m2) * L1 * a_w1 * a_w1 * sin_d
                  - (m1 + m2) * g * sin(a_th2)) / den2

        # k4
        a_th1 = th1 + dt * k3_th1
        a_w1 = w1 + dt * k3_w1
        a_th2 = th2 + dt * k3_th2
        a_w2 = w2 + dt * k3_w2
        delta = a_th2 - a_th1
        sin_d = sin(delta)
        cos_d = cos(delta)
        den1 = (m1 + m2) * L1 - m2 * L1 * cos_d * cos_d
        den2 = (L2 / L1) * den1
        k4_th1 = a_w1
        k4_w1 = (m2 * L1 * a_w1 * a_w1 * sin_d * cos_d
                  + m2 * g * sin(a_th2) * cos_d
                  + m2 * L2 * a_w2 * a_w2 * sin_d
                  - (m1 + m2) * g * sin(a_th1)) / den1
        k4_th2 = a_w2
        k4_w2 = (-m2 * L2 * a_w2 * a_w2 * sin_d * cos_d
                  + (m1 + m2) * g * sin(a_th1) * cos_d
                  - (m1 + m2) * L1 * a_w1 * a_w1 * sin_d
                  - (m1 + m2) * g * sin(a_th2)) / den2

        # Update
        th1 += dt * (k1_th1 + 2.0 * k2_th1 + 2.0 * k3_th1 + k4_th1) / 6.0
        w1 += dt * (k1_w1 + 2.0 * k2_w1 + 2.0 * k3_w1 + k4_w1) / 6.0
        th2 += dt * (k1_th2 + 2.0 * k2_th2 + 2.0 * k3_th2 + k4_th2) / 6.0
        w2 += dt * (k1_w2 + 2.0 * k2_w2 + 2.0 * k3_w2 + k4_w2) / 6.0

    return th1
