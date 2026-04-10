def approx_e(n=40):
    """Approximate Euler's number e using the Taylor series sum of 1/k!.

    Returns the approximation after n terms.
    """
    total = 0.0
    factorial = 1.0
    for k in range(1, n + 1):
        factorial *= k
        total += 1.0 / factorial
    return 1.0 + total
