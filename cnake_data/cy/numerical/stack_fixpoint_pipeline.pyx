# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Run fixed-point arithmetic pipeline and summarize results (Cython).

Adapted from The Stack v2 Cython candidate:
- blob_id: 459af010814400430dd7b6b4b11abf116ef11993
- filename: fixpoint.pyx
"""

from cnake_data.benchmarks import cython_benchmark


cdef int FIX_SHIFT = 16
cdef int FIX_ONE = 1 << 16


cdef inline int _int2fix(int v) noexcept nogil:
    return v << FIX_SHIFT


cdef inline int _mul(int a, int b) noexcept nogil:
    return <int>((<long long>a * b) >> FIX_SHIFT)


cdef inline int _div(int a, int b) noexcept nogil:
    return <int>(((<long long>a) << FIX_SHIFT) // b)


cdef int _sqrt_fix(int x) noexcept nogil:
    cdef int xn, nxt, t
    if x == 0:
        return 0
    xn = _int2fix(1)
    for t in range(12):
        if xn == 0:
            break
        nxt = (xn + _div(x, xn)) // 2
        if nxt == xn:
            break
        xn = nxt
    return xn


@cython_benchmark(syntax="cy", args=(12000,))
def stack_fixpoint_pipeline(int n):
    cdef int acc = _int2fix(3)
    cdef int step = _int2fix(2)
    cdef unsigned int root_acc = 0
    cdef int i, x, r

    for i in range(1, n + 1):
        x = _int2fix((i % 17) + 1)
        acc = _mul(acc + x, step)
        acc = acc - _int2fix(i % 5)
        if acc <= 0:
            acc = _int2fix(1)
        r = _sqrt_fix(acc)
        root_acc = (root_acc + <unsigned int>(r + i)) & 0xFFFFFFFF
        acc = acc % _int2fix(997)

    return (acc, root_acc, acc >> FIX_SHIFT, (root_acc >> 8) & 0xFFFF)
