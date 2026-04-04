"""Golden section search for minimizing n different unimodal functions.

Each function is a shifted quadratic with deterministic coefficients.
Returns summary of minima found.

Keywords: optimization, golden section, search, minimization, unimodal, benchmark
"""

import math

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(5000,))
def golden_section(n: int) -> tuple:
    """Find minima of n unimodal functions using golden section search.

    Function i: f_i(x) = a_i*(x - c_i)^2 + b_i*sin(x - c_i) where
    a_i, b_i, c_i are deterministic from i.

    Args:
        n: Number of functions to minimize.

    Returns:
        Tuple of (sum_minima, min_of_first, min_of_last).
    """
    gr = (math.sqrt(5.0) + 1.0) / 2.0
    max_iter = 200
    tol = 1e-12

    sum_minima = 0.0
    min_of_first = 0.0
    min_of_last = 0.0

    for idx in range(n):
        # Deterministic coefficients
        a = 1.0 + (idx % 10) * 0.5
        b = 0.3 * math.sin(idx * 0.1)
        c = -5.0 + 10.0 * ((idx * 7 + 3) % n) / max(n, 1)

        # Search interval around c
        lo = c - 10.0
        hi = c + 10.0

        for _ in range(max_iter):
            if hi - lo < tol:
                break
            d = (hi - lo) / gr
            x1 = hi - d
            x2 = lo + d

            f1 = a * (x1 - c) * (x1 - c) + b * math.sin(x1 - c)
            f2 = a * (x2 - c) * (x2 - c) + b * math.sin(x2 - c)

            if f1 < f2:
                hi = x2
            else:
                lo = x1

        x_min = (lo + hi) / 2.0
        f_min = a * (x_min - c) * (x_min - c) + b * math.sin(x_min - c)
        sum_minima += f_min

        if idx == 0:
            min_of_first = f_min
        if idx == n - 1:
            min_of_last = f_min

    return (sum_minima, min_of_first, min_of_last)
