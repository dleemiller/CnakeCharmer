# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Monin-Obukhov stability length batch computation — Cython implementation."""

from libc.math cimport atan, exp, fabs, log, pi, sqrt

from cnake_data.benchmarks import cython_benchmark


cdef double _psim(double zeta) nogil:
    cdef double x, tmp
    if zeta <= 0.0:
        tmp = sqrt(1.0 - 16.0 * zeta)
        x = sqrt(tmp)
        return pi / 2.0 - 2.0 * atan(x) + log((1.0 + x) * (1.0 + x) * (1.0 + x * x) / 8.0)
    return -2.0 / 3.0 * (zeta - 5.0 / 0.35) * exp(-0.35 * zeta) - zeta - (10.0 / 3.0) / 0.35


cdef double _psih(double zeta) nogil:
    cdef double x, tmp, v
    if zeta <= 0.0:
        tmp = sqrt(1.0 - 16.0 * zeta)
        x = sqrt(tmp)
        return 2.0 * log((1.0 + x * x) / 2.0)
    v = 1.0 + (2.0 / 3.0) * zeta
    return (
        -2.0 / 3.0 * (zeta - 5.0 / 0.35) * exp(-0.35 * zeta)
        - v * sqrt(v)
        - (10.0 / 3.0) / 0.35
        + 1.0
    )


cdef double _ribtol(double rib, double zsl, double z0m, double z0h) nogil:
    cdef double L, L0, Ls, Le, fx, fxs, fxe, fxdif, ds, de
    L = 1.0 if rib > 0.0 else -1.0
    L0 = 2.0 * L

    while fabs(L - L0) > 0.001:
        L0 = L
        ds = (log(zsl / z0m) - _psim(zsl / L) + _psim(z0m / L))
        fx = rib - zsl / L * (log(zsl / z0h) - _psih(zsl / L) + _psih(z0h / L)) / (ds * ds)

        Ls = L - 0.001 * L
        Le = L + 0.001 * L
        ds = (log(zsl / z0m) - _psim(zsl / Ls) + _psim(z0m / Ls))
        de = (log(zsl / z0m) - _psim(zsl / Le) + _psim(z0m / Le))
        fxs = -zsl / Ls * (log(zsl / z0h) - _psih(zsl / Ls) + _psih(z0h / Ls)) / (ds * ds)
        fxe = -zsl / Le * (log(zsl / z0h) - _psih(zsl / Le) + _psih(z0h / Le)) / (de * de)
        fxdif = (fxe - fxs) / (0.002 * L)

        if fxdif != 0.0:
            L = L - fx / fxdif
        else:
            break
        if L > 1e4:
            L = 1e4
        elif L < -1e4:
            L = -1e4

    return L


@cython_benchmark(syntax="cy", args=(500,))
def monin_obukhov_stability(int n):
    """Compute Obukhov length L for n bulk Richardson numbers in [-0.5, 0.5].

    Args:
        n: Number of Rib values to process.

    Returns:
        Tuple of (mean_L, first_L, last_L).
    """
    cdef double zsl = 10.0
    cdef double z0m = 0.1
    cdef double z0h = 0.01
    cdef double total = 0.0
    cdef double first_L = 0.0
    cdef double last_L = 0.0
    cdef double rib, L
    cdef int i

    for i in range(n):
        rib = -0.5 + i * (1.0 / (n - 1)) if n > 1 else 0.0
        if fabs(rib) < 0.01:
            rib = 0.01
        L = _ribtol(rib, zsl, z0m, z0h)
        total += L
        if i == 0:
            first_L = L
        last_L = L

    return (total / n, first_L, last_L)
