"""Evaluate a polynomial at multiple points using Horner's method.

Keywords: numerical, polynomial, evaluation, horner, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(5000,))
def polynomial_eval(n: int) -> float:
    """Evaluate a degree-n polynomial at n points using Horner's method.

    Coefficients: c[i] = (i*7+3)%100 / 10.0.
    Points: x[j] = (j*13+7)%1000 / 1000.0 (range 0..~1).
    Returns sum of all evaluations.

    Args:
        n: Degree of polynomial and number of evaluation points.

    Returns:
        Sum of all polynomial evaluations as a float.
    """
    coeffs = [(i * 7 + 3) % 100 / 10.0 for i in range(n)]
    points = [(j * 13 + 7) % 1000 / 1000.0 for j in range(n)]

    total = 0.0
    for x in points:
        # Horner's method: evaluate from highest degree down
        result = coeffs[n - 1]
        for k in range(n - 2, -1, -1):
            result = result * x + coeffs[k]
        total += result

    return total
