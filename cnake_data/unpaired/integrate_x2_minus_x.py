def integrate_x2_minus_x(a, b, n=2000):
    """Integrate f(x) = x^2 - x from a to b using the rectangle rule.

    Returns the approximate integral.
    """
    dx = (b - a) / n
    s = 0.0
    for i in range(n):
        x = a + i * dx
        s += x * x - x
    return s * dx
