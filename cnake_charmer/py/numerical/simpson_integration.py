"""Simpson's rule integration of f(x) = sin(x) * exp(-x/100) from 0 to 10.

Keywords: numerical, integration, Simpson's rule, trigonometry, benchmark
"""

import math

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(10000000,))
def simpson_integration(n: int) -> float:
    """Integrate f(x) = sin(x) * exp(-x/100) from 0 to 10 using Simpson's rule.

    Uses n intervals (must be even; if odd, n is incremented by 1).

    Args:
        n: Number of intervals.

    Returns:
        Approximate value of the integral.
    """
    if n % 2 == 1:
        n += 1

    a = 0.0
    b = 10.0
    h = (b - a) / n

    result = math.sin(a) * math.exp(-a / 100.0) + math.sin(b) * math.exp(-b / 100.0)

    for i in range(1, n, 2):
        x = a + i * h
        result += 4.0 * math.sin(x) * math.exp(-x / 100.0)

    for i in range(2, n, 2):
        x = a + i * h
        result += 2.0 * math.sin(x) * math.exp(-x / 100.0)

    return result * h / 3.0
