"""Sin-squared accumulation over a synthetic grid.

Keywords: numerical, trigonometry, sin, sum, benchmark
"""

import math

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(0.0, 0.001, 50000))
def sin_squared_sum(offset: float, step: float, samples: int) -> float:
    """Accumulate sin(x)^2 over evenly spaced samples."""
    total = 0.0
    for i in range(samples):
        x = offset + i * step
        s = math.sin(x)
        total += s * s
    return total
