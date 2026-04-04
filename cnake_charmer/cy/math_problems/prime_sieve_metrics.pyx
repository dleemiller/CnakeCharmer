# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Compute prime sieve summary metrics with modular accumulation (Cython)."""

from libc.stdlib cimport malloc, free

from cnake_charmer.benchmarks import cython_benchmark


cdef void _sieve_and_metrics(
    unsigned char *is_prime,
    int limit,
    int window,
    int mod_base,
    int *count_out,
    int *checksum_out,
    int *max_gap_out,
) noexcept nogil:
    cdef int i, j, n, step, gap
    cdef int count = 0
    cdef int checksum = 0
    cdef int max_gap = 0
    cdef int prev = 2

    for i in range(limit + 1):
        is_prime[i] = 1
    is_prime[0] = 0
    is_prime[1] = 0

    i = 2
    while i * i <= limit:
        if is_prime[i]:
            step = i
            j = i * i
            while j <= limit:
                is_prime[j] = 0
                j += step
        i += 1

    for n in range(2, limit + 1):
        if is_prime[n]:
            count += 1
            checksum = (checksum + (n % mod_base) * ((n % window) + 1)) % mod_base
            gap = n - prev
            if gap > max_gap:
                max_gap = gap
            prev = n

    count_out[0] = count
    checksum_out[0] = checksum
    max_gap_out[0] = max_gap


@cython_benchmark(syntax="cy", args=(400000, 128, 1000003))
def prime_sieve_metrics(int limit, int window, int mod_base):
    cdef unsigned char *is_prime
    cdef int count = 0
    cdef int checksum = 0
    cdef int max_gap = 0

    if limit < 2:
        return (0, 0, 0)

    is_prime = <unsigned char *>malloc((limit + 1) * sizeof(unsigned char))
    if not is_prime:
        raise MemoryError()

    with nogil:
        _sieve_and_metrics(is_prime, limit, window, mod_base, &count, &checksum, &max_gap)

    free(is_prime)
    return (count, checksum, max_gap)
