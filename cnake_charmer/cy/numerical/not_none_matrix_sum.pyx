# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Sum traces of dot products from Matrix pairs with not-None checks.

Keywords: numerical, matrix, not none, extension type, trace, cython, benchmark
"""

from cnake_charmer.benchmarks import cython_benchmark


cdef class Matrix:
    """Simple 2x2 matrix stored as four doubles."""
    cdef double a, b, c, d

    def __cinit__(self, double a, double b,
                  double c, double d):
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def dot(self, Matrix other not None):
        """Multiply self by other, return new Matrix."""
        return Matrix(
            self.a * other.a + self.b * other.c,
            self.a * other.b + self.b * other.d,
            self.c * other.a + self.d * other.c,
            self.c * other.b + self.d * other.d,
        )

    cdef double trace(self):
        return self.a + self.d


@cython_benchmark(syntax="cy", args=(10000,))
def not_none_matrix_sum(int n):
    """Create n matrix pairs, multiply, sum traces."""
    cdef double total = 0.0
    cdef int i
    cdef double a, b, c, d, a2, b2, c2, d2
    cdef Matrix m1, m2, result

    for i in range(n):
        a = ((<long long>i * <long long>2654435761)
             % 1000) / 100.0
        b = ((<long long>i * <long long>1664525
              + <long long>1013904223) % 1000) / 100.0
        c = ((<long long>i * <long long>1103515245
              + 12345) % 1000) / 100.0
        d = ((<long long>i * <long long>214013
              + <long long>2531011) % 1000) / 100.0
        m1 = Matrix(a, b, c, d)

        a2 = ((<long long>i
               * <long long>1566083941
               + 1) % 1000) / 100.0
        b2 = ((<long long>i
               * <long long>2053540636
               + 7) % 1000) / 100.0
        c2 = ((<long long>i
               * <long long>1654435769
               + 13) % 1000) / 100.0
        d2 = ((<long long>i
               * <long long>1013904243
               + 19) % 1000) / 100.0
        m2 = Matrix(a2, b2, c2, d2)

        result = m1.dot(m2)
        total += result.trace()
    return total
