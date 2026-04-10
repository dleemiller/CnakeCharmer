import math


def sqrt_trial_prime(n):
    """Check if n is prime using trial division up to sqrt(n).

    Tests divisibility by 2, then odd numbers from 3 to sqrt(n).
    Returns True if n is prime, False otherwise.
    """
    if n == 2:
        return True
    if n % 2 == 0 or n <= 1:
        return False
    search_up_to = int(math.sqrt(n)) + 1
    return all(n % divisor != 0 for divisor in range(3, search_up_to, 2))
