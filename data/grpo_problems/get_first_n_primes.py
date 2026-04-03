def get_first_n_primes(n):
    """Find the first n prime numbers using trial division.

    Tests each candidate integer for divisibility by all previously
    found primes.  Returns a list of the first n primes.
    """
    primes = [0] * n
    i = 0
    candidate = 1

    while i < n:
        candidate += 1
        is_ok = True
        for c in range(i):
            if candidate % primes[c] == 0:
                is_ok = False
                break
        if is_ok:
            primes[i] = candidate
            i += 1

    return primes
