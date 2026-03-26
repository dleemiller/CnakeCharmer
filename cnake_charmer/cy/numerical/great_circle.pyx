# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Great circle distance computation (Cython-optimized).

Keywords: great circle, haversine, distance, geography, numerical, cython, benchmark
"""

from libc.math cimport acos, cos, sin, M_PI
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(100000,))
def great_circle(int n):
    """Compute sum of great circle distances for n point pairs using C math."""
    cdef double radius = 3956.0
    cdef double pi_180 = M_PI / 180.0
    cdef double total = 0.0
    cdef double lat1, lon1, lat2, lon2
    cdef double a, b, theta, c
    cdef int i

    for i in range(n):
        lat1 = ((i * 7 + 3) % 180 - 90) * pi_180
        lon1 = ((i * 13 + 7) % 360 - 180) * pi_180
        lat2 = ((i * 11 + 5) % 180 - 90) * pi_180
        lon2 = ((i * 17 + 11) % 360 - 180) * pi_180

        a = (M_PI / 2.0) - lat1
        b = (M_PI / 2.0) - lat2
        theta = lon2 - lon1

        c = acos(cos(a) * cos(b) + sin(a) * sin(b) * cos(theta))
        total += radius * c

    return total
