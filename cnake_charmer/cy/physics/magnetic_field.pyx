# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Compute magnetic field from current-carrying wire segments using Biot-Savart law (Cython-optimized).

Keywords: physics, magnetic field, Biot-Savart, electromagnetism, wire, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from libc.math cimport sqrt, sin, cos, M_PI
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(800,))
def magnetic_field(int n):
    """Compute magnetic field at observation points from n wire segments."""
    cdef double mu0_over_4pi = 1e-7
    cdef double current = 1.0
    cdef int n_obs = 200
    cdef int n_sub = 10
    cdef int obs, seg, sub, idx
    cdef double ox, oy, oz, bx, by, bz
    cdef double sx0, sy0, sx1, sy1, t0, t1, tmid
    cdef double dlx, dly, dlz, mx, my, mz
    cdef double rx, ry, rz, r2, r, r3
    cdef double cx, cy_val, cz, coeff
    cdef double total_bx = 0.0
    cdef double total_by = 0.0
    cdef double total_bz = 0.0

    # Precompute wire vertices
    cdef double *wx = <double *>malloc((n + 1) * sizeof(double))
    cdef double *wy = <double *>malloc((n + 1) * sizeof(double))
    if not wx or not wy:
        if wx: free(wx)
        if wy: free(wy)
        raise MemoryError()

    cdef double angle
    for idx in range(n + 1):
        angle = 2.0 * M_PI * idx / n
        wx[idx] = cos(angle)
        wy[idx] = sin(angle)

    for obs in range(n_obs):
        ox = -2.0 + 4.0 * obs / (n_obs - 1)
        oy = 0.0
        oz = 0.5

        bx = 0.0
        by = 0.0
        bz = 0.0

        for seg in range(n):
            sx0 = wx[seg]
            sy0 = wy[seg]
            sx1 = wx[seg + 1]
            sy1 = wy[seg + 1]

            for sub in range(n_sub):
                t0 = <double>sub / n_sub
                t1 = <double>(sub + 1) / n_sub
                tmid = 0.5 * (t0 + t1)

                dlx = (sx1 - sx0) / n_sub
                dly = (sy1 - sy0) / n_sub
                dlz = 0.0

                mx = sx0 + tmid * (sx1 - sx0)
                my = sy0 + tmid * (sy1 - sy0)
                mz = 0.0

                rx = ox - mx
                ry = oy - my
                rz = oz - mz

                r2 = rx * rx + ry * ry + rz * rz
                r = sqrt(r2)
                if r < 1e-15:
                    continue
                r3 = r2 * r

                cx = dly * rz - dlz * ry
                cy_val = dlz * rx - dlx * rz
                cz = dlx * ry - dly * rx

                coeff = mu0_over_4pi * current / r3
                bx += coeff * cx
                by += coeff * cy_val
                bz += coeff * cz

        total_bx += bx
        total_by += by
        total_bz += bz

    free(wx)
    free(wy)
    return (total_bx, total_by, total_bz)
