def distinct_prime_factors(n):
    """Count numbers up to n with exactly 2 distinct prime factors.

    Returns (count, last_found, total_prime_factors_sum).
    """
    count = 0
    last_found = 0
    total_pf_sum = 0

    for num in range(2, n + 1):
        factors = set()
        temp = num
        d = 2
        while d * d <= temp:
            while temp % d == 0:
                factors.add(d)
                temp //= d
            d += 1
        if temp > 1:
            factors.add(temp)

        total_pf_sum += len(factors)
        if len(factors) == 2:
            count += 1
            last_found = num

    return (count, last_found, total_pf_sum)
