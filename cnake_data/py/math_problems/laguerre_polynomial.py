"""Evaluate generalized Laguerre polynomials iteratively over many x values.

Keywords: laguerre, polynomial, numerical, iterative, orthogonal, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(50000,))
def laguerre_polynomial(n: int) -> tuple:
    """Evaluate L_k^alpha(x) for k=15, alpha=0.5 at n evenly-spaced x in [0, 20].

    Uses the three-term recurrence relation for generalized Laguerre polynomials:
        L_0 = 1
        L_1 = -x + alpha + 1
        L_i = ((2i - 1 + alpha - x) / i) * L_{i-1} - ((i + alpha - 1) / i) * L_{i-2}

    Args:
        n: Number of evaluation points.

    Returns:
        Tuple of (sum of all polynomial values, value at the midpoint x=10).
    """
    k = 15
    alpha = 0.5
    total = 0.0
    mid_val = 0.0
    mid_idx = n // 2

    for idx in range(n):
        x = 20.0 * idx / (n - 1) if n > 1 else 0.0

        # Evaluate L_k^alpha(x) via three-term recurrence
        minus_2 = 1.0
        minus_1 = -x + alpha + 1.0
        for i in range(2, k + 1):
            a = (2.0 * i - 1.0 + alpha - x) / i
            b = (i + alpha - 1.0) / i
            current = a * minus_1 - b * minus_2
            minus_2 = minus_1
            minus_1 = current

        val = minus_1
        total += val
        if idx == mid_idx:
            mid_val = val

    return (total, mid_val)
