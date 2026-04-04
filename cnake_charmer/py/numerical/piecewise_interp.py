"""
Piecewise linear interpolation (AFGEN algorithm).

Keywords: numerical, interpolation, piecewise, linear, afgen, benchmark
"""

import math

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(5000,))
def piecewise_interp(n: int) -> tuple:
    """Evaluate piecewise linear interpolation at m=n query points.

    x[i] = i * 1.0, y[i] = sin(2*pi*i/n)*100 for i in 0..n-1.
    Query points: t[j] = j * (n-1) / (n*1.0) for j in 0..n-1.

    AFGEN: find bracket [x[i-1], x[i]] where x[i-1] <= t < x[i].

    Args:
        n: Number of data points and query points.

    Returns:
        (int(sum_results * 1000) % 10**9, int(results[n//3] * 1e6))
    """
    two_pi = 2.0 * math.pi

    # Build x and y arrays
    x = [float(i) for i in range(n)]
    y = [math.sin(two_pi * i / n) * 100.0 for i in range(n)]

    # Evaluate at m=n query points
    results = [0.0] * n
    for j in range(n):
        t = j * (n - 1) / (n * 1.0)

        # Edge cases
        if t <= x[0]:
            results[j] = y[0]
        elif t >= x[n - 1]:
            results[j] = y[n - 1]
        else:
            # Find bracket: binary search for x[i-1] <= t < x[i]
            # Since x[i] = i, bracket index i = int(t) + 1 (when t is not integer)
            # Linear: x[i] = i so bracket is floor(t)+1
            i = int(t) + 1
            if i >= n:
                i = n - 1
            x0 = x[i - 1]
            x1 = x[i]
            y0 = y[i - 1]
            y1 = y[i]
            alpha = (t - x0) / (x1 - x0)
            results[j] = y0 + alpha * (y1 - y0)

    sum_results = sum(results)
    return (int(sum_results * 1000) % (10**9), int(results[n // 3] * 1e6))
