# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Nelder-Mead simplex optimization (Cython-optimized).

Keywords: nelder-mead, simplex, optimization, direct search, cython, benchmark
"""

from libc.math cimport sin
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(10000,))
def simplex_nelder_mead(int n):
    """Nelder-Mead optimization of f(x,y) = (x-1)^2 + (y-2)^2 + sin(x*y)."""
    cdef double total = 0.0
    cdef int idx, it
    cdef double x0, y0
    cdef double vx0, vy0, vx1, vy1, vx2, vy2
    cdef double f0, f1, f2
    cdef double cx, cy, rx, ry, fr, ex, ey, fe
    cdef double kx, ky, fk, best

    for idx in range(n):
        x0 = (idx * 0.7) % 5 - 2.5
        y0 = (idx * 1.3) % 5 - 2.5

        vx0 = x0
        vy0 = y0
        vx1 = x0 + 1.0
        vy1 = y0
        vx2 = x0
        vy2 = y0 + 1.0

        f0 = (vx0 - 1.0) * (vx0 - 1.0) + (vy0 - 2.0) * (vy0 - 2.0) + sin(vx0 * vy0)
        f1 = (vx1 - 1.0) * (vx1 - 1.0) + (vy1 - 2.0) * (vy1 - 2.0) + sin(vx1 * vy1)
        f2 = (vx2 - 1.0) * (vx2 - 1.0) + (vy2 - 2.0) * (vy2 - 2.0) + sin(vx2 * vy2)

        for it in range(100):
            # Sort
            if f0 > f1:
                vx0, vx1 = vx1, vx0
                vy0, vy1 = vy1, vy0
                f0, f1 = f1, f0
            if f0 > f2:
                vx0, vx2 = vx2, vx0
                vy0, vy2 = vy2, vy0
                f0, f2 = f2, f0
            if f1 > f2:
                vx1, vx2 = vx2, vx1
                vy1, vy2 = vy2, vy1
                f1, f2 = f2, f1

            cx = (vx0 + vx1) * 0.5
            cy = (vy0 + vy1) * 0.5

            rx = 2.0 * cx - vx2
            ry = 2.0 * cy - vy2
            fr = (rx - 1.0) * (rx - 1.0) + (ry - 2.0) * (ry - 2.0) + sin(rx * ry)

            if fr < f1:
                if fr < f0:
                    ex = 3.0 * cx - 2.0 * vx2
                    ey = 3.0 * cy - 2.0 * vy2
                    fe = (ex - 1.0) * (ex - 1.0) + (ey - 2.0) * (ey - 2.0) + sin(ex * ey)
                    if fe < fr:
                        vx2 = ex
                        vy2 = ey
                        f2 = fe
                    else:
                        vx2 = rx
                        vy2 = ry
                        f2 = fr
                else:
                    vx2 = rx
                    vy2 = ry
                    f2 = fr
            else:
                if fr < f2:
                    kx = 0.5 * (cx + rx)
                    ky = 0.5 * (cy + ry)
                else:
                    kx = 0.5 * (cx + vx2)
                    ky = 0.5 * (cy + vy2)
                fk = (kx - 1.0) * (kx - 1.0) + (ky - 2.0) * (ky - 2.0) + sin(kx * ky)

                if fk < f2:
                    vx2 = kx
                    vy2 = ky
                    f2 = fk
                else:
                    vx1 = vx0 + 0.5 * (vx1 - vx0)
                    vy1 = vy0 + 0.5 * (vy1 - vy0)
                    vx2 = vx0 + 0.5 * (vx2 - vx0)
                    vy2 = vy0 + 0.5 * (vy2 - vy0)
                    f1 = (vx1 - 1.0) * (vx1 - 1.0) + (vy1 - 2.0) * (vy1 - 2.0) + sin(vx1 * vy1)
                    f2 = (vx2 - 1.0) * (vx2 - 1.0) + (vy2 - 2.0) * (vy2 - 2.0) + sin(vx2 * vy2)

        best = f0
        if f1 < best:
            best = f1
        if f2 < best:
            best = f2
        total += best

    return total
