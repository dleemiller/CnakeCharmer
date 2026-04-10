import math


def quad_integrate_sinx2(a, b, n):
    """Integrate sin(x^2) from a to b using the rectangle rule with n steps.

    Returns the approximate integral value.
    """
    dx = float(b - a) / n
    s = 0.0
    for i in range(n):
        x = a + i * dx
        s += math.sin(x * x)
    return s * dx
