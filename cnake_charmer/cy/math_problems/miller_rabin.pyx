# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Count primes up to n using deterministic Miller-Rabin (Cython-optimized).

Keywords: math, primes, miller-rabin, primality, number theory, cython, benchmark
"""

from cnake_charmer.benchmarks import cython_benchmark


cdef long long mod_pow(long long base, long long exp, long long mod):
    """Fast modular exponentiation."""
    cdef long long result = 1
    base = base % mod
    while exp > 0:
        if exp & 1:
            result = result * base % mod
        exp >>= 1
        base = base * base % mod
    return result


cdef int is_prime_mr(long long num):
    """Deterministic Miller-Rabin primality test."""
    cdef long long d, x, a
    cdef int r, i, j
    cdef int witnesses[12]

    if num < 2:
        return 0
    if num < 4:
        return 1
    if num % 2 == 0 or num % 3 == 0:
        return 0

    d = num - 1
    r = 0
    while d % 2 == 0:
        d = d / 2
        r += 1

    witnesses[0] = 2
    witnesses[1] = 3
    witnesses[2] = 5
    witnesses[3] = 7
    witnesses[4] = 11
    witnesses[5] = 13
    witnesses[6] = 17
    witnesses[7] = 19
    witnesses[8] = 23
    witnesses[9] = 29
    witnesses[10] = 31
    witnesses[11] = 37

    for i in range(12):
        a = witnesses[i]
        if a >= num:
            continue
        x = mod_pow(a, d, num)
        if x == 1 or x == num - 1:
            continue
        for j in range(r - 1):
            x = mod_pow(x, 2, num)
            if x == num - 1:
                break
        else:
            return 0
    return 1


@cython_benchmark(syntax="cy", args=(100000,))
def miller_rabin(int n):
    """Count primes up to n using typed Miller-Rabin."""
    cdef int i, count
    cdef long long num

    if n < 2:
        return 0

    count = 0
    for i in range(2, n + 1):
        num = i
        if is_prime_mr(num):
            count += 1

    return count
