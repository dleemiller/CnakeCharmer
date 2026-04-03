# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Fanno flow compressible gas dynamics.

Keywords: fanno flow, compressible flow, gas dynamics, mach number, thermodynamics, cython
"""

from libc.math cimport sqrt, log, fabs

from cnake_charmer.benchmarks import cython_benchmark


cdef double p_pstar(double ma, double gamma) nogil:
    cdef double gm1 = gamma - 1
    cdef double gp1 = gamma + 1
    cdef double m2 = ma * ma
    cdef double factor = 1.0 + gm1 / 2.0 * m2
    return (1.0 / ma) * sqrt(gp1 / 2.0) / sqrt(factor)


cdef double t_tstar(double ma, double gamma) nogil:
    cdef double gm1 = gamma - 1
    cdef double gp1 = gamma + 1
    cdef double m2 = ma * ma
    return gp1 / (2.0 * (1.0 + gm1 / 2.0 * m2))


cdef double rho_rhostar(double ma, double gamma) nogil:
    cdef double gm1 = gamma - 1
    cdef double gp1 = gamma + 1
    cdef double m2 = ma * ma
    return (1.0 / ma) * sqrt((2.0 / gp1) * (1.0 + gm1 / 2.0 * m2))


cdef double nondim_length(double ma, double gamma) nogil:
    cdef double m2 = ma * ma
    cdef double gp1 = gamma + 1
    cdef double gm1 = gamma - 1
    return (1 - m2) / (gamma * m2) + gp1 / (2 * gamma) * log(
        gp1 * m2 / (2 * (1 + gm1 / 2 * m2)))


cdef double fanno_ma_from_length(double l_param, double gamma, double ma0) nogil:
    cdef double x1 = ma0
    cdef double x2 = ma0 + 0.01
    cdef double x3, f1, f2
    cdef int it

    for it in range(100):
        f1 = nondim_length(x1, gamma) - l_param
        f2 = nondim_length(x2, gamma) - l_param
        if fabs(f2) < 1e-12:
            break
        if fabs(f2 - f1) < 1e-15:
            break
        x3 = x2 - f2 * (x2 - x1) / (f2 - f1)
        x1 = x2
        x2 = x3

    return x2


@cython_benchmark(syntax="cy", args=(2000,))
def fanno_flow(int n):
    """Compute Fanno flow properties for n Mach numbers.

    Args:
        n: Number of Mach number evaluation points.

    Returns:
        Tuple of (total_pressure_ratio, total_temp_ratio, total_length_param).
    """
    cdef double gamma = 1.4
    cdef double total_p = 0.0
    cdef double total_t = 0.0
    cdef double total_l = 0.0
    cdef double ma, p_ratio, t_ratio, r_ratio, l_param, ma_inv, ma0
    cdef double step
    cdef int i

    step = 2.95 / (n - 1) if n > 1 else 0.0

    with nogil:
        for i in range(n):
            if n > 1:
                ma = 0.05 + i * step
            else:
                ma = 1.0

            p_ratio = p_pstar(ma, gamma)
            t_ratio = t_tstar(ma, gamma)
            r_ratio = rho_rhostar(ma, gamma)

            total_p += p_ratio
            total_t += t_ratio * r_ratio

            if ma < 0.99 or ma > 1.01:
                l_param = nondim_length(ma, gamma)
                if ma < 1.0:
                    ma0 = 0.1
                else:
                    ma0 = 1.5
                ma_inv = fanno_ma_from_length(l_param, gamma, ma0)
                total_l += fabs(ma - ma_inv)

    return (total_p, total_t, total_l)
