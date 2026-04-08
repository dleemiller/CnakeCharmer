# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""1D Kalman filter for scalar state estimation — Cython implementation."""

from libc.math cimport sin

from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(5000,))
def kalman_filter_1d(int n_steps):
    """Run a 1D Kalman filter on a deterministic noisy measurement sequence."""
    cdef double Q = 0.01, R = 1.0, A = 1.0, H = 1.0
    cdef double x_est = 0.0, P = 1.0
    cdef double sum_x = 0.0, sum_innov = 0.0, gain = 0.0
    cdef double true_x, z, noise, x_pred, P_pred, S, innov
    cdef unsigned long lcg = 12345
    cdef unsigned long lcg_a = 1664525
    cdef unsigned long lcg_c = 1013904223
    cdef unsigned long lcg_m = 4294967296  # 2**32
    cdef int k

    for k in range(n_steps):
        true_x = sin(k * 0.02)
        lcg = (lcg_a * lcg + lcg_c) % lcg_m
        noise = (<double>lcg / <double>lcg_m - 0.5) * 2.0
        z = true_x + noise
        x_pred = A * x_est
        P_pred = A * P * A + Q
        S = H * P_pred * H + R
        gain = P_pred * H / S
        innov = z - H * x_pred
        x_est = x_pred + gain * innov
        P = (1.0 - gain * H) * P_pred
        sum_x += x_est
        sum_innov += innov

    return (sum_x, sum_innov, gain)
