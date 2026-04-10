import math


def numerical_integrate(a, b, n=2000):
    """Numerical integration of sin(x^2) from a to b using the rectangle rule.

    Divides [a, b] into n intervals and sums f(x)*dx.
    Returns the approximate integral.
    """
    dx = float(b - a) / n
    s = 0.0
    for i in range(n):
        x = a + i * dx
        s += math.sin(x * x)
    return s * dx
