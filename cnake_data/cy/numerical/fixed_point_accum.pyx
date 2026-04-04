# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Accumulate fixed-point arithmetic values with mixed operations using a cdef class.

Keywords: fixed-point, arithmetic, accumulate, cdef class, __add__, __sub__, __mul__, __iadd__, __neg__, __bool__, cython, benchmark
"""

from cnake_data.benchmarks import cython_benchmark


cdef class FixedPoint:
    cdef long long _val

    def __init__(self, long long val):
        self._val = val

    def __add__(x, y):
        cdef FixedPoint a = <FixedPoint>x
        cdef FixedPoint b = <FixedPoint>y
        return FixedPoint(a._val + b._val)

    def __sub__(x, y):
        cdef FixedPoint a = <FixedPoint>x
        cdef FixedPoint b = <FixedPoint>y
        return FixedPoint(a._val - b._val)

    def __mul__(x, y):
        cdef FixedPoint a = <FixedPoint>x
        cdef FixedPoint b = <FixedPoint>y
        # Multiply and rescale: (a*S) * (b*S) / S = a*b*S
        # Floor division matching Python semantics (cdivision truncates)
        cdef long long prod = a._val * b._val
        cdef long long q = prod // 1000
        # Correct C truncation to Python floor division for negative values
        if prod < 0 and q * 1000 != prod:
            q -= 1
        return FixedPoint(q)

    def __iadd__(self, other):
        cdef FixedPoint b = <FixedPoint>other
        self._val += b._val
        return self

    def __neg__(self):
        return FixedPoint(-self._val)

    def __bool__(self):
        return self._val != 0


@cython_benchmark(syntax="cy", args=(100000,))
def fixed_point_accum(int n):
    """Accumulate n fixed-point values with mixed arithmetic operations.

    Generates deterministic fixed-point numbers and combines them using
    __add__, __sub__, __mul__, __iadd__, __neg__, and __bool__.

    Args:
        n: Number of values to accumulate.

    Returns:
        Final integer value (scaled by 1000).
    """
    cdef FixedPoint accum = FixedPoint(1000)  # Start at 1.000
    cdef FixedPoint fp, small, neg_fp
    cdef int i
    cdef unsigned long long h1, h2
    cdef long long raw
    cdef int op

    for i in range(n):
        h1 = ((<unsigned long long>i * <unsigned long long>2654435761 + 1) >> 8) & 0xFFFF
        h2 = ((<unsigned long long>i * <unsigned long long>1103515245 + 3) >> 8) & 0xFFFF

        # Generate a fixed-point value in range [-5.000, 5.000]
        raw = <long long>(h1 % 10001) - 5000
        fp = FixedPoint(raw)

        # Choose operation based on h2
        op = h2 % 5

        if op == 0:
            # __add__
            accum = accum + fp
        elif op == 1:
            # __sub__
            accum = accum - fp
        elif op == 2:
            # __mul__ (scale the value to keep it small)
            # Multiply by a value near 1.0: 0.990 to 1.010
            small = FixedPoint(1000 + <long long>(h1 % 21) - 10)
            accum = accum * small
        elif op == 3:
            # __iadd__
            accum += fp
        else:
            # __neg__ and __bool__
            neg_fp = -fp
            if neg_fp:
                accum = accum + neg_fp
            else:
                accum = accum + FixedPoint(1000)

    return accum._val
