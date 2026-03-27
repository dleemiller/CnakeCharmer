"""Fused element-wise expression a*b + c*d using NumPy.

Keywords: numerical, fused expression, numpy, benchmark
"""

import numpy as np

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(1000000,))
def pythran_fused_expr(n: int) -> float:
    """Evaluate a*b + c*d element-wise and return sum.

    NumPy creates 3 temporary arrays (a*b, c*d, sum of those).

    Args:
        n: Array length.

    Returns:
        Sum of the fused expression.
    """
    rng = np.random.RandomState(42)
    a = rng.standard_normal(n)
    b = rng.standard_normal(n)
    c = rng.standard_normal(n)
    d = rng.standard_normal(n)
    result = np.sum(a * b + c * d)
    return float(result)
