"""Newton-Raphson root finding for polynomial x^3 - 2x - 1.

Keywords: numerical, newton-raphson, root finding, polynomial, fractal, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(50000,))
def newton_raphson_poly(n: int) -> tuple:
    """Find roots of x^3 - 2x - 1 using Newton-Raphson from many starting points.

    For n evenly spaced starting points in [-3.0, 3.0], run Newton-Raphson
    iteration and record converged root values. Returns checksum of all
    converged roots and count of points that converged.

    Args:
        n: Number of starting points.

    Returns:
        Tuple of (checksum, converge_count).
    """
    checksum = 0.0
    converge_count = 0
    step = 6.0 / (n - 1) if n > 1 else 0.0

    for i in range(n):
        x = -3.0 + i * step

        for _ in range(50):
            fx = x * x * x - 2.0 * x - 1.0
            fpx = 3.0 * x * x - 2.0
            if fpx == 0.0:
                break
            x_new = x - fx / fpx
            if abs(x_new - x) < 1e-10:
                x = x_new
                break
            x = x_new

        checksum += x

        fx_final = x * x * x - 2.0 * x - 1.0
        if abs(fx_final) < 1e-8:
            converge_count += 1

    return (checksum, converge_count)
