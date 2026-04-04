"""Piecewise linear interpolation from a table of (x, y) breakpoints.

Given n query points spread across the breakpoint range, find the bracketing
interval via linear search and interpolate.  Returns (sum, max) of all
interpolated values.

Keywords: numerical, interpolation, piecewise linear, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(100000,))
def piecewise_interp(n: int) -> tuple:
    """Evaluate piecewise linear interpolation at n query points.

    Args:
        n: Number of query points.

    Returns:
        Tuple of (total_sum, max_val) of interpolated values.
    """
    # Build a table of 50 breakpoints
    num_bp = 50
    x_bp = [0.0] * num_bp
    y_bp = [0.0] * num_bp
    for i in range(num_bp):
        x_bp[i] = i * 2.0
        y_bp[i] = ((i * i * 7 + 3) % 100) / 10.0

    total_sum = 0.0
    max_val = -1e300

    for i in range(n):
        t = (i * 97.0 / n) % 98.0

        # Clamp below
        if t <= x_bp[0]:
            val = y_bp[0]
        # Clamp above
        elif t >= x_bp[num_bp - 1]:
            val = y_bp[num_bp - 1]
        else:
            # Linear search for bracketing interval
            j = 0
            while j < num_bp - 1 and x_bp[j + 1] < t:
                j += 1
            x1 = x_bp[j]
            x2 = x_bp[j + 1]
            y1 = y_bp[j]
            y2 = y_bp[j + 1]
            val = y1 + (y2 - y1) / (x2 - x1) * (t - x1)

        total_sum += val
        if val > max_val:
            max_val = val

    return (total_sum, max_val)
