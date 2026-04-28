def trial_division_primes(n):
    """Return all primes in [2, n) via trial division."""
    result = []
    for i in range(2, n):
        j = 2
        is_prime = True
        while j * j <= i:
            if i % j == 0:
                is_prime = False
                break
            j += 1
        if is_prime:
            result.append(i)
    return result
