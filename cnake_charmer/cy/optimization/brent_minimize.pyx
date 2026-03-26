# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Brent's method for function minimization (Cython-optimized).

Keywords: brent, minimization, golden section, optimization, cython, benchmark
"""

from libc.math cimport fabs
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(100000,))
def brent_minimize(int n):
    """Find minima of f_i(x) = x^4 - 3x^2 + x + i*0.001 using Brent's method."""
    cdef double golden = 0.3819660112501051
    cdef double total = 0.0
    cdef int idx, it
    cdef double offset, a, b, x, w, v, fx, fw, fv, e, d
    cdef double midpoint, tol1, tol2, r, q, p_val, u, fu

    for idx in range(n):
        offset = idx * 0.001
        a = -3.0
        b = 3.0

        x = 0.0
        w = 0.0
        v = 0.0
        fx = x * x * x * x - 3.0 * x * x + x + offset
        fw = fx
        fv = fx
        e = 0.0
        d = 0.0

        for it in range(50):
            midpoint = 0.5 * (a + b)
            tol1 = 1e-8 * fabs(x) + 1e-10
            tol2 = 2.0 * tol1

            if fabs(x - midpoint) <= (tol2 - 0.5 * (b - a)):
                break

            if fabs(e) > tol1:
                r = (x - w) * (fx - fv)
                q = (x - v) * (fx - fw)
                p_val = (x - v) * q - (x - w) * r
                q = 2.0 * (q - r)
                if q > 0.0:
                    p_val = -p_val
                else:
                    q = -q
                if fabs(p_val) < fabs(0.5 * q * e) and p_val > q * (a - x) and p_val < q * (b - x):
                    e = d
                    d = p_val / q
                else:
                    if x < midpoint:
                        e = b - x
                    else:
                        e = a - x
                    d = golden * e
            else:
                if x < midpoint:
                    e = b - x
                else:
                    e = a - x
                d = golden * e

            if fabs(d) >= tol1:
                u = x + d
            else:
                if d > 0:
                    u = x + tol1
                else:
                    u = x - tol1

            fu = u * u * u * u - 3.0 * u * u + u + offset

            if fu <= fx:
                if u < x:
                    b = x
                else:
                    a = x
                v = w
                fv = fw
                w = x
                fw = fx
                x = u
                fx = fu
            else:
                if u < x:
                    a = u
                else:
                    b = u
                if fu <= fw or w == x:
                    v = w
                    fv = fw
                    w = u
                    fw = fu
                elif fu <= fv or v == x or v == w:
                    v = u
                    fv = fu

        total += fx

    return total
