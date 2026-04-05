# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Profile mixed integer tools: gcd/lcm/sieve/nth-prime and divisor counts (Cython).

Adapted from The Stack v2 Cython candidate:
- blob_id: 97834639626f5afbe569e7cf7799a8e08b01b9ea
- filename: inttools.pyx
"""

from libc.stdlib cimport malloc, free
from cnake_data.benchmarks import cython_benchmark


cdef inline int _gcd(int a, int b) noexcept nogil:
    cdef int t
    while b:
        t = a % b
        a = b
        b = t
    return a


@cython_benchmark(syntax="cy", args=(120000,))
def stack_inttools_profile(int n):
    cdef unsigned int g = 0
    cdef long long l = 1
    cdef int mod = 1000000007
    cdef int i, a, b, d
    cdef int lim = n
    cdef unsigned char *is_prime
    cdef int pcount = 0
    cdef int nth_idx
    cdef int nth = 0
    cdef int j
    cdef int x, r, c
    cdef int div_sum = 0

    for i in range(2, 2 + n // 6):
        a = i * 37 + 11
        b = i * 53 + 7
        d = _gcd(a, b)
        g ^= <unsigned int>d
        l = (l * (((a // d) * b) % mod)) % mod

    if lim < 2:
        lim = 2
    is_prime = <unsigned char *>malloc((lim + 1) * sizeof(unsigned char))
    if not is_prime:
        raise MemoryError()
    for i in range(lim + 1):
        is_prime[i] = 1
    is_prime[0] = 0
    is_prime[1] = 0

    i = 2
    while i * i <= lim:
        if is_prime[i]:
            j = i * i
            while j <= lim:
                is_prime[j] = 0
                j += i
        i += 1

    nth_idx = n // 30
    for i in range(2, lim + 1):
        if is_prime[i]:
            if pcount == nth_idx:
                nth = i
            pcount += 1

    if nth == 0:
        for i in range(lim, 1, -1):
            if is_prime[i]:
                nth = i
                break

    free(is_prime)

    for x in range((n - 80) if n > 82 else 2, n):
        c = 0
        r = 1
        while r * r <= x:
            if x % r == 0:
                if r * r == x:
                    c += 1
                else:
                    c += 2
            r += 1
        div_sum += c

    return (g & 0xFFFFFFFF, l, nth, div_sum)
