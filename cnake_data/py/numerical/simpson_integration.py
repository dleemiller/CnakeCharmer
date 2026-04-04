"""Simpson's rule integration of sin(x)*exp(-x/n) over [0, n].

Keywords: numerical, integration, Simpson's rule, trigonometry, benchmark
"""

import math

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(3000000,))
def simpson_integration(n: int) -> tuple:
    """Integrate f(x) = sin(x) * exp(-x/n) from 0 to n using Simpson's rule.

    Args:
        n: Both the integration upper bound and the number of panels.

    Returns:
        Tuple of (integral, midpoint_contrib, num_panels).
        midpoint_contrib is the function value at x = n/2.
    """
    panels = n
    if panels % 2 == 1:
        panels += 1

    a = 0.0
    b = float(n)
    h = (b - a) / panels

    result = math.sin(a) * math.exp(-a / n) + math.sin(b) * math.exp(-b / n)

    for i in range(1, panels, 2):
        x = a + i * h
        result += 4.0 * math.sin(x) * math.exp(-x / n)

    for i in range(2, panels, 2):
        x = a + i * h
        result += 2.0 * math.sin(x) * math.exp(-x / n)

    integral = result * h / 3.0

    mid_x = b / 2.0
    midpoint_contrib = math.sin(mid_x) * math.exp(-mid_x / n)

    return (integral, midpoint_contrib, panels)
