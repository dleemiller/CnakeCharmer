def factorial_recursive(n):
    """Recursive factorial."""
    if n <= 1:
        return 1.0
    return n * factorial_recursive(n - 1)


def factorial_iterative(n):
    """Iterative factorial."""
    fac = 1.0
    for i in range(2, n + 1):
        fac *= i
    return fac


def a_n(n):
    """Compute (2^n * (n!)^2) / (2n)! ."""
    fn = factorial_recursive(n)
    return (2**n) * (fn * fn) / factorial_recursive(2 * n)
