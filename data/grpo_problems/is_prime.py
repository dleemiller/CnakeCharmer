def is_prime(n):
    """Check if n is prime by trial division up to n//2.

    Returns True if n is prime, False otherwise.
    """
    if n < 2:
        return False
    top = n // 2
    return all(n % i != 0 for i in range(2, top))
