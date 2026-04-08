"""Newton's method (Heron's method) square root over a range.

Computes the sum of square roots for integers 1..n using the iterative
Babylonian/Newton's method: x_{k+1} = (x_k + v/x_k) / 2.

Keywords: numerical, newton, square root, iteration, convergence, babylonian
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(5000,))
def newton_sqrt_sum(n: int) -> tuple:
    """Sum of Newton's-method square roots for integers 1..n.

    Each sqrt is computed via Heron's iterative method starting from 1.0,
    iterating until convergence within 1e-14 or 100 iterations.

    Args:
        n: Upper bound (inclusive).

    Returns:
        Tuple of (total_sum, midpoint_sqrt, last_sqrt).
    """
    total = 0.0
    mid_val = 0.0
    last_val = 0.0
    mid = n // 2
    for v in range(1, n + 1):
        x = 1.0
        fv = float(v)
        for _ in range(100):
            xprev = x
            x = (x + fv / x) * 0.5
            diff = xprev - x if x - xprev < 0.0 else x - xprev
            if diff < 1e-14:
                break
        total += x
        if v == mid:
            mid_val = x
        last_val = x
    return (total, mid_val, last_val)
