# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Compute divmod of hash-derived pairs and accumulate sums
(Cython-optimized with ctuple return).

Keywords: ctuple, divmod, math, cython, benchmark
"""

from cnake_data.benchmarks import cython_benchmark


cdef (long long, long long) c_divmod(
    long long a, long long b
):
    """Compute quotient and remainder as ctuple."""
    return (a / b, a % b)


@cython_benchmark(syntax="cy", args=(100000,))
def ctuple_divmod(int n):
    """Compute divmod of n pairs, return q_sum + r_sum."""
    cdef long long q_sum = 0
    cdef long long r_sum = 0
    cdef int i
    cdef unsigned long long h1, h2
    cdef long long a, b
    cdef (long long, long long) qr

    for i in range(n):
        h1 = (<unsigned long long>i
              * <unsigned long long>2654435761)
        h2 = (<unsigned long long>i
              * <unsigned long long>2246822519)
        a = <long long>(
            (h1 & <unsigned long long>0xFFFFFFFF)
            % 1000000 + 1
        )
        b = <long long>(
            (h2 & <unsigned long long>0xFFFFFFFF)
            % 999 + 1
        )
        qr = c_divmod(a, b)
        q_sum += qr[0]
        r_sum += qr[1]
    return q_sum + r_sum
