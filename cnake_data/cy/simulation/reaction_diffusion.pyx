# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""1D reaction-diffusion simulation (Gray-Scott model, Cython-optimized).

Keywords: simulation, reaction-diffusion, gray-scott, PDE, pattern formation, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from libc.math cimport exp
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(1000,))
def reaction_diffusion(int n):
    """Simulate 1D Gray-Scott reaction-diffusion with C arrays."""
    cdef int steps = 1000
    cdef double du_coeff = 0.16
    cdef double dv_coeff = 0.08
    cdef double feed = 0.035
    cdef double kill = 0.065
    cdef double dt = 1.0
    cdef double sigma = n / 10.0
    cdef int i, t, im1, ip1
    cdef double dist_sq, gauss, lap_u, lap_v, uvv, total

    cdef double *u_arr = <double *>malloc(n * sizeof(double))
    cdef double *v_arr = <double *>malloc(n * sizeof(double))
    cdef double *u_new = <double *>malloc(n * sizeof(double))
    cdef double *v_new = <double *>malloc(n * sizeof(double))
    cdef double *tmp

    if not u_arr or not v_arr or not u_new or not v_new:
        free(u_arr); free(v_arr); free(u_new); free(v_new)
        raise MemoryError()

    # Initialize
    for i in range(n):
        dist_sq = (i - n / 2.0) * (i - n / 2.0)
        gauss = exp(-dist_sq / (sigma * sigma))
        u_arr[i] = 1.0 - gauss
        v_arr[i] = gauss

    for t in range(steps):
        for i in range(n):
            im1 = (i - 1 + n) % n
            ip1 = (i + 1) % n
            lap_u = u_arr[im1] - 2.0 * u_arr[i] + u_arr[ip1]
            lap_v = v_arr[im1] - 2.0 * v_arr[i] + v_arr[ip1]
            uvv = u_arr[i] * v_arr[i] * v_arr[i]
            u_new[i] = u_arr[i] + dt * (du_coeff * lap_u - uvv + feed * (1.0 - u_arr[i]))
            v_new[i] = v_arr[i] + dt * (dv_coeff * lap_v + uvv - (feed + kill) * v_arr[i])
        tmp = u_arr
        u_arr = u_new
        u_new = tmp
        tmp = v_arr
        v_arr = v_new
        v_new = tmp

    total = 0.0
    for i in range(n):
        total += v_arr[i]

    free(u_arr)
    free(v_arr)
    free(u_new)
    free(v_new)
    return total
