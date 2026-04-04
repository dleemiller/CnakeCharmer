# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Solve y''=-2*y+x BVP with y(0)=0, y(1)=1 using shooting method (Cython-optimized).

Try shooting slopes and bisect to find correct one. Return y(0.5).
n = grid resolution for each ODE solve. 10 bisection iterations.

Keywords: BVP, shooting method, boundary value problem, bisection, ODE, cython
"""

from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(100000,))
def shooting_method(int n):
    """Solve y''=-2*y+x BVP using shooting method and return y(0.5)."""
    cdef int i, k, half_n
    cdef double dt, slope_lo, slope_hi, slope_mid
    cdef double val_lo, val_hi, val_mid
    cdef double y1, y2, x, k1_y1, k1_y2, k2_y1, k2_y2
    cdef double k3_y1, k3_y2, k4_y1, k4_y2, best_slope, half_dt

    dt = 1.0 / n
    half_dt = 0.5 * dt
    half_n = n // 2

    # Evaluate at slope_lo = 0.0
    slope_lo = 0.0
    y1 = 0.0
    y2 = slope_lo
    x = 0.0
    for i in range(n):
        k1_y1 = dt * y2
        k1_y2 = dt * (-2.0 * y1 + x)
        k2_y1 = dt * (y2 + 0.5 * k1_y2)
        k2_y2 = dt * (-2.0 * (y1 + 0.5 * k1_y1) + x + half_dt)
        k3_y1 = dt * (y2 + 0.5 * k2_y2)
        k3_y2 = dt * (-2.0 * (y1 + 0.5 * k2_y1) + x + half_dt)
        k4_y1 = dt * (y2 + k3_y2)
        k4_y2 = dt * (-2.0 * (y1 + k3_y1) + x + dt)
        y1 += (k1_y1 + 2.0 * k2_y1 + 2.0 * k3_y1 + k4_y1) / 6.0
        y2 += (k1_y2 + 2.0 * k2_y2 + 2.0 * k3_y2 + k4_y2) / 6.0
        x += dt
    val_lo = y1 - 1.0

    # Evaluate at slope_hi = 3.0
    slope_hi = 3.0
    y1 = 0.0
    y2 = slope_hi
    x = 0.0
    for i in range(n):
        k1_y1 = dt * y2
        k1_y2 = dt * (-2.0 * y1 + x)
        k2_y1 = dt * (y2 + 0.5 * k1_y2)
        k2_y2 = dt * (-2.0 * (y1 + 0.5 * k1_y1) + x + half_dt)
        k3_y1 = dt * (y2 + 0.5 * k2_y2)
        k3_y2 = dt * (-2.0 * (y1 + 0.5 * k2_y1) + x + half_dt)
        k4_y1 = dt * (y2 + k3_y2)
        k4_y2 = dt * (-2.0 * (y1 + k3_y1) + x + dt)
        y1 += (k1_y1 + 2.0 * k2_y1 + 2.0 * k3_y1 + k4_y1) / 6.0
        y2 += (k1_y2 + 2.0 * k2_y2 + 2.0 * k3_y2 + k4_y2) / 6.0
        x += dt
    val_hi = y1 - 1.0

    # 10 bisection steps
    for k in range(10):
        slope_mid = 0.5 * (slope_lo + slope_hi)
        y1 = 0.0
        y2 = slope_mid
        x = 0.0
        for i in range(n):
            k1_y1 = dt * y2
            k1_y2 = dt * (-2.0 * y1 + x)
            k2_y1 = dt * (y2 + 0.5 * k1_y2)
            k2_y2 = dt * (-2.0 * (y1 + 0.5 * k1_y1) + x + half_dt)
            k3_y1 = dt * (y2 + 0.5 * k2_y2)
            k3_y2 = dt * (-2.0 * (y1 + 0.5 * k2_y1) + x + half_dt)
            k4_y1 = dt * (y2 + k3_y2)
            k4_y2 = dt * (-2.0 * (y1 + k3_y1) + x + dt)
            y1 += (k1_y1 + 2.0 * k2_y1 + 2.0 * k3_y1 + k4_y1) / 6.0
            y2 += (k1_y2 + 2.0 * k2_y2 + 2.0 * k3_y2 + k4_y2) / 6.0
            x += dt
        val_mid = y1 - 1.0
        if val_lo * val_mid <= 0:
            slope_hi = slope_mid
            val_hi = val_mid
        else:
            slope_lo = slope_mid
            val_lo = val_mid

    # Use best slope to compute y at x=0.5
    best_slope = 0.5 * (slope_lo + slope_hi)
    y1 = 0.0
    y2 = best_slope
    x = 0.0
    for i in range(half_n):
        k1_y1 = dt * y2
        k1_y2 = dt * (-2.0 * y1 + x)
        k2_y1 = dt * (y2 + 0.5 * k1_y2)
        k2_y2 = dt * (-2.0 * (y1 + 0.5 * k1_y1) + x + half_dt)
        k3_y1 = dt * (y2 + 0.5 * k2_y2)
        k3_y2 = dt * (-2.0 * (y1 + 0.5 * k2_y1) + x + half_dt)
        k4_y1 = dt * (y2 + k3_y2)
        k4_y2 = dt * (-2.0 * (y1 + k3_y1) + x + dt)
        y1 += (k1_y1 + 2.0 * k2_y1 + 2.0 * k3_y1 + k4_y1) / 6.0
        y2 += (k1_y2 + 2.0 * k2_y2 + 2.0 * k3_y2 + k4_y2) / 6.0
        x += dt

    return y1
