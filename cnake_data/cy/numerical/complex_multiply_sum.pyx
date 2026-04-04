# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Multiply complex numbers and sum the results using a cdef class.

Keywords: complex, multiply, arithmetic, cdef class, __add__, __mul__, operator overloading, cython, benchmark
"""

from cnake_data.benchmarks import cython_benchmark


cdef class Complex:
    cdef double re
    cdef double im

    def __init__(self, double re, double im):
        self.re = re
        self.im = im

    def __add__(x, y):
        cdef Complex a = <Complex>x
        cdef Complex b = <Complex>y
        return Complex(a.re + b.re, a.im + b.im)

    def __mul__(x, y):
        cdef Complex a = <Complex>x
        cdef Complex b = <Complex>y
        return Complex(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re)


@cython_benchmark(syntax="cy", args=(100000,))
def complex_multiply_sum(int n):
    """Generate n complex number pairs, multiply them, and sum all products.

    Args:
        n: Number of complex multiplications.

    Returns:
        Tuple of (real_sum, imag_sum).
    """
    cdef double total_re = 0.0
    cdef double total_im = 0.0
    cdef int i
    cdef unsigned long long h1, h2, h3, h4
    cdef double a_re, a_im, b_re, b_im
    cdef double prod_re, prod_im

    for i in range(n):
        h1 = ((<unsigned long long>i * <unsigned long long>2654435761 + 1) >> 8) & 0xFFFF
        h2 = ((<unsigned long long>i * <unsigned long long>1103515245 + 3) >> 8) & 0xFFFF
        a_re = (<int>(h1 % 201) - 100) / 10.0
        a_im = (<int>(h2 % 201) - 100) / 10.0

        h3 = ((<unsigned long long>i * <unsigned long long>6364136223846793005 + 7) >> 16) & 0xFFFF
        h4 = ((<unsigned long long>i * <unsigned long long>3935559000370003845 + 11) >> 16) & 0xFFFF
        b_re = (<int>(h3 % 201) - 100) / 10.0
        b_im = (<int>(h4 % 201) - 100) / 10.0

        # Complex multiply: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        prod_re = a_re * b_re - a_im * b_im
        prod_im = a_re * b_im + a_im * b_re

        # Accumulate
        total_re += prod_re
        total_im += prod_im

    return (total_re, total_im)
