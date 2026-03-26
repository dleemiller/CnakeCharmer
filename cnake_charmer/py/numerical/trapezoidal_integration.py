"""
Numerical integration of f(x)=x^2 from 0 to 1 using the trapezoidal rule.

Keywords: integration, trapezoidal, numerical, calculus, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(1000000,))
def trapezoidal_integration(n: int) -> float:
    """Integrate f(x)=x^2 from 0 to 1 using the trapezoidal rule with n steps.

    Args:
        n: Number of trapezoids (steps).

    Returns:
        Approximate value of the integral as a float.
    """
    a = 0.0
    b = 1.0
    h = (b - a) / n

    result = 0.5 * (a * a + b * b)
    for i in range(1, n):
        x = a + i * h
        result += x * x

    return result * h
