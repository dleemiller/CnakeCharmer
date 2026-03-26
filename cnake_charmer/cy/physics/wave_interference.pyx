# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Compute interference pattern of point sources (Cython-optimized).

Keywords: physics, wave, interference, diffraction, amplitude, cython, benchmark
"""

from libc.math cimport sin, sqrt, M_PI
from libc.stdlib cimport malloc, free
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(500,))
def wave_interference(int n):
    """Compute wave interference with C arrays and typed nested loops."""
    cdef double wavelength = 1.0
    cdef double two_pi_over_lam = 2.0 * M_PI / wavelength
    cdef int n_obs = 1000
    cdef double total_intensity = 0.0
    cdef double obs_x, obs_y, amplitude, src_x, dx, r
    cdef int obs, i

    cdef double *src_xs = <double *>malloc(n * sizeof(double))
    if not src_xs:
        raise MemoryError()

    for i in range(n):
        src_xs[i] = i * 0.5

    obs_y = 10.0
    for obs in range(n_obs):
        obs_x = obs * 0.01 - 5.0
        amplitude = 0.0
        for i in range(n):
            dx = obs_x - src_xs[i]
            r = sqrt(dx * dx + obs_y * obs_y)
            if r > 0:
                amplitude += sin(two_pi_over_lam * r) / r
        total_intensity += amplitude * amplitude

    free(src_xs)
    return total_intensity
