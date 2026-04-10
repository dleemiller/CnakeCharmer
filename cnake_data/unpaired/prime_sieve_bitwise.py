def prime_sieve_bitwise(n):
    """Count primes up to n using a simple sieve with bitwise tricks.

    Returns (prime_count, sum_of_primes_mod_1000000007, largest_prime).
    """
    if n < 2:
        return (0, 0, 0)

    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False

    p = 2
    while p * p <= n:
        if is_prime[p]:
            for i in range(p * p, n + 1, p):
                is_prime[i] = False
        p += 1

    count = 0
    prime_sum = 0
    largest = 0
    mod = 1000000007

    for i in range(2, n + 1):
        if is_prime[i]:
            count += 1
            prime_sum = (prime_sum + i) % mod
            largest = i

    return (count, prime_sum, largest)
