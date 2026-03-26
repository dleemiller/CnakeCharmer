"""Solve y''=-2*y+x BVP with y(0)=0, y(1)=1 using shooting method.

Try shooting slopes and bisect to find correct one. Return y(0.5).
n = grid resolution for each ODE solve. 10 bisection iterations.

Keywords: BVP, shooting method, boundary value problem, bisection, ODE
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(100000,))
def shooting_method(n: int) -> float:
    """Solve y''=-2*y+x BVP using shooting method and return y(0.5).

    Args:
        n: Grid resolution for each ODE solve.

    Returns:
        Value of y at x=0.5.
    """
    dt = 1.0 / n

    def solve_ivp(slope):
        """Solve y''=-2*y+x as system: y1'=y2, y2'=-2*y1+x. y1(0)=0, y2(0)=slope."""
        y1 = 0.0
        y2 = slope
        x = 0.0
        for _ in range(n):
            # RK4 for the system
            k1_y1 = dt * y2
            k1_y2 = dt * (-2.0 * y1 + x)
            k2_y1 = dt * (y2 + 0.5 * k1_y2)
            k2_y2 = dt * (-2.0 * (y1 + 0.5 * k1_y1) + x + 0.5 * dt)
            k3_y1 = dt * (y2 + 0.5 * k2_y2)
            k3_y2 = dt * (-2.0 * (y1 + 0.5 * k2_y1) + x + 0.5 * dt)
            k4_y1 = dt * (y2 + k3_y2)
            k4_y2 = dt * (-2.0 * (y1 + k3_y1) + x + dt)
            y1 += (k1_y1 + 2.0 * k2_y1 + 2.0 * k3_y1 + k4_y1) / 6.0
            y2 += (k1_y2 + 2.0 * k2_y2 + 2.0 * k3_y2 + k4_y2) / 6.0
            x += dt
        return y1

    def solve_at_half(slope):
        """Solve and return y at x=0.5."""
        y1 = 0.0
        y2 = slope
        x = 0.0
        half_n = n // 2
        for _ in range(half_n):
            k1_y1 = dt * y2
            k1_y2 = dt * (-2.0 * y1 + x)
            k2_y1 = dt * (y2 + 0.5 * k1_y2)
            k2_y2 = dt * (-2.0 * (y1 + 0.5 * k1_y1) + x + 0.5 * dt)
            k3_y1 = dt * (y2 + 0.5 * k2_y2)
            k3_y2 = dt * (-2.0 * (y1 + 0.5 * k2_y1) + x + 0.5 * dt)
            k4_y1 = dt * (y2 + k3_y2)
            k4_y2 = dt * (-2.0 * (y1 + k3_y1) + x + dt)
            y1 += (k1_y1 + 2.0 * k2_y1 + 2.0 * k3_y1 + k4_y1) / 6.0
            y2 += (k1_y2 + 2.0 * k2_y2 + 2.0 * k3_y2 + k4_y2) / 6.0
            x += dt
        return y1

    # Bisection: find slope s such that solve_ivp(s) = 1.0 (target y(1)=1)
    slope_lo = 0.0
    slope_hi = 3.0

    val_lo = solve_ivp(slope_lo) - 1.0

    # 10 bisection iterations
    for _k in range(10):
        slope_mid = 0.5 * (slope_lo + slope_hi)
        val_mid = solve_ivp(slope_mid) - 1.0
        if val_lo * val_mid <= 0:
            slope_hi = slope_mid
        else:
            slope_lo = slope_mid
            val_lo = val_mid

    best_slope = 0.5 * (slope_lo + slope_hi)
    return solve_at_half(best_slope)
