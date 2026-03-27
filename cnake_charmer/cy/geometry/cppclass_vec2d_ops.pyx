# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
# distutils: language = c++
"""2D vector arithmetic using an inline C++ Vec2d class with operator overloading.

Keywords: 2D vectors, dot product, cppclass, cdef extern, geometry, benchmark
"""

from libc.stdlib cimport malloc, free
from libc.math cimport sin, cos
from cnake_charmer.benchmarks import cython_benchmark

cdef extern from *:
    """
    #include <cmath>
    struct Vec2d {
        double x, y;
        Vec2d() : x(0.0), y(0.0) {}
        Vec2d(double x_, double y_) : x(x_), y(y_) {}
        Vec2d operator+(const Vec2d& o) const { return Vec2d(x + o.x, y + o.y); }
        double dot(const Vec2d& o) const { return x * o.x + y * o.y; }
    };
    """
    cdef cppclass Vec2d:
        double x
        double y
        Vec2d()
        Vec2d(double x_, double y_)
        Vec2d operator+(Vec2d o)
        double dot(Vec2d o)


@cython_benchmark(syntax="cy", args=(300000,))
def cppclass_vec2d_ops(int n):
    """Generate n 2D vectors and compute aggregate statistics.

    Vector i: x_i = sin(i * 0.1), y_i = cos(i * 0.1).
    Accumulate all vectors to get a sum vector.
    Compute dot products of consecutive pairs (i, i+1) and sum them.

    Args:
        n: Number of vectors to generate.

    Returns:
        Tuple of (magnitude_squared_of_sum, sum_of_dot_products).
    """
    cdef double *xs = <double *>malloc(n * sizeof(double))
    cdef double *ys = <double *>malloc(n * sizeof(double))
    if not xs or not ys:
        if xs: free(xs)
        if ys: free(ys)
        raise MemoryError()

    cdef int i
    cdef double xi, yi
    cdef double acc_x = 0.0
    cdef double acc_y = 0.0
    cdef double dot_sum = 0.0

    for i in range(n):
        xi = sin(i * 0.1)
        yi = cos(i * 0.1)
        xs[i] = xi
        ys[i] = yi
        acc_x += xi
        acc_y += yi

    for i in range(n - 1):
        dot_sum += xs[i] * xs[i + 1] + ys[i] * ys[i + 1]

    cdef double mag_sq = acc_x * acc_x + acc_y * acc_y

    free(xs)
    free(ys)

    return (mag_sq, dot_sum)
