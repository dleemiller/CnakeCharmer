"""Linear interpolation using NumPy.

Uses np.interp to interpolate values at query points.

Keywords: numerical, interpolation, linear, numpy, benchmark
"""

import numpy as np

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(100000,))
def numpy_typed_interp(n: int) -> float:
    """Interpolate n query points and return sum.

    Args:
        n: Number of query points.

    Returns:
        Sum of interpolated values.
    """
    rng = np.random.RandomState(42)
    num_knots = 1000
    xp = np.linspace(0.0, 1.0, num_knots)
    fp = np.cumsum(rng.standard_normal(num_knots))
    x_query = rng.random(n)
    result = np.interp(x_query, xp, fp)
    return float(np.sum(result))
