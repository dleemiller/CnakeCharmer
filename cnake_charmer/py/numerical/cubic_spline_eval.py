"""Evaluate a natural cubic spline at n midpoints.

Knots at x=0,1,...,n with y[i]=(i*7+3)%100. Evaluates the spline at the
midpoint of each interval and returns the sum of interpolated values.

Keywords: numerical, interpolation, cubic spline, tridiagonal, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(50000,))
def cubic_spline_eval(n: int) -> float:
    """Evaluate natural cubic spline at n midpoints and return their sum.

    Args:
        n: Number of intervals (n+1 knots).

    Returns:
        Sum of interpolated values at midpoints.
    """
    # Knot values
    y = [(i * 7 + 3) % 100 for i in range(n + 1)]

    # Natural cubic spline: solve tridiagonal system for second derivatives
    # h[i] = 1 for all intervals (uniform spacing)
    # System: h[i-1]*M[i-1] + 2*(h[i-1]+h[i])*M[i] + h[i]*M[i+1] = 6*(d[i]-d[i-1])
    # With h=1: M[i-1] + 4*M[i] + M[i+1] = 6*(y[i+1] - 2*y[i] + y[i-1])

    m = n - 1  # number of interior points
    if m <= 0:
        return float(y[0])

    # Right-hand side
    rhs = [0.0] * m
    for i in range(m):
        rhs[i] = 6.0 * (y[i + 2] - 2.0 * y[i + 1] + y[i])

    # Tridiagonal solve: diag=4, off-diag=1
    # Forward elimination
    c_prime = [0.0] * m
    d_prime = [0.0] * m

    c_prime[0] = 1.0 / 4.0
    d_prime[0] = rhs[0] / 4.0

    for i in range(1, m):
        denom = 4.0 - c_prime[i - 1]
        c_prime[i] = 1.0 / denom
        d_prime[i] = (rhs[i] - d_prime[i - 1]) / denom

    # Back substitution
    second_deriv = [0.0] * (n + 1)  # M[0]=M[n]=0 for natural spline
    second_deriv[m] = d_prime[m - 1]
    for i in range(m - 2, -1, -1):
        second_deriv[i + 1] = d_prime[i] - c_prime[i] * second_deriv[i + 2]

    # Evaluate at midpoints
    total = 0.0
    for i in range(n):
        t = 0.5  # midpoint of interval [i, i+1], h=1
        mi = second_deriv[i]
        mi1 = second_deriv[i + 1]
        yi = y[i]
        yi1 = y[i + 1]
        # Cubic spline formula with h=1
        val = (mi * (1.0 - t) ** 3 + mi1 * t**3) / 6.0
        val += (yi - mi / 6.0) * (1.0 - t) + (yi1 - mi1 / 6.0) * t
        total += val

    return total
