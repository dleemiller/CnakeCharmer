# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Multi-layer Snell's law refraction through a planar optical stack — Cython implementation."""

from libc.math cimport fabs, pi, sin, sqrt

from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(3000, 10))
def snell_refraction(int n_rays, int n_layers):
    """Trace n_rays through n_layers planar interfaces using Snell's law."""
    cdef double layer_d = 5.0
    cdef double sum_exit_sin = 0.0, total_path = 0.0
    cdef int transmitted = 0
    cdef double theta, sin_theta, sin_theta2, cos_t, path_len
    cdef double n1, n2
    cdef int i, layer
    cdef bint tir

    for i in range(n_rays):
        if n_rays > 1:
            theta = pi / 4.0 * i / (n_rays - 1)
        else:
            theta = 0.0
        sin_theta = sin(theta)
        path_len = 0.0
        tir = False

        for layer in range(n_layers):
            # Alternating air (1.0) and glass (1.5)
            n1 = 1.0 if layer % 2 == 0 else 1.5
            n2 = 1.5 if layer % 2 == 0 else 1.0
            sin_theta2 = n1 * sin_theta / n2
            if fabs(sin_theta2) > 1.0:
                tir = True
                break
            cos_t = sqrt(1.0 - sin_theta2 * sin_theta2)
            if cos_t > 0.0:
                path_len += layer_d / cos_t
            sin_theta = sin_theta2

        if not tir:
            sum_exit_sin += sin_theta
            transmitted += 1
            total_path += path_len

    return (sum_exit_sin, transmitted, total_path)
