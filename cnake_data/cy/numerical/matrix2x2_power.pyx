# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Chain-multiply n 2x2 matrices and return the trace using a cdef class.

Keywords: matrix, 2x2, multiply, chain, trace, cdef class, __mul__, cython, benchmark
"""

from cnake_data.benchmarks import cython_benchmark


cdef class Mat2x2:
    cdef double a
    cdef double b
    cdef double c
    cdef double d

    def __init__(self, double a, double b, double c, double d):
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def __mul__(x, y):
        cdef Mat2x2 s = <Mat2x2>x
        cdef Mat2x2 o = <Mat2x2>y
        return Mat2x2(
            s.a * o.a + s.b * o.c,
            s.a * o.b + s.b * o.d,
            s.c * o.a + s.d * o.c,
            s.c * o.b + s.d * o.d,
        )

    cdef double trace(self):
        return self.a + self.d


@cython_benchmark(syntax="cy", args=(100000,))
def matrix2x2_power(int n):
    """Generate n 2x2 matrices and chain-multiply them, returning the trace.

    Each matrix has small entries to avoid overflow. The chain product
    M_0 * M_1 * ... * M_{n-1} is computed left-to-right using __mul__.

    Args:
        n: Number of matrices to chain-multiply.

    Returns:
        Trace of the resulting product matrix.
    """
    cdef Mat2x2 result = Mat2x2(1.0, 0.0, 0.0, 1.0)
    cdef Mat2x2 m
    cdef int i
    cdef unsigned long long h1, h2, h3, h4
    cdef double ma, mb, mc, md

    for i in range(n):
        h1 = ((<unsigned long long>i * <unsigned long long>2654435761 + 1) >> 8) & 0xFFFF
        h2 = ((<unsigned long long>i * <unsigned long long>1103515245 + 3) >> 8) & 0xFFFF
        h3 = ((<unsigned long long>i * <unsigned long long>2246822519 + 5) >> 8) & 0xFFFF
        h4 = ((<unsigned long long>i * <unsigned long long>6364136223846793005 + 7) >> 16) & 0xFFFF

        # Small entries near identity to keep values bounded
        ma = 1.0 + (<int>(h1 % 21) - 10) / 1000.0
        mb = (<int>(h2 % 21) - 10) / 1000.0
        mc = (<int>(h3 % 21) - 10) / 1000.0
        md = 1.0 + (<int>(h4 % 21) - 10) / 1000.0

        m = Mat2x2(ma, mb, mc, md)
        result = result * m

    return result.trace()
