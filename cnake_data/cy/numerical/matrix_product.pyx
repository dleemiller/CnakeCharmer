# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Product of n distinct 2x2 integer matrices mod M (Cython-optimized).

Keywords: matrix, product, modular arithmetic, numerical, cython, benchmark
"""

from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(500000,))
def matrix_product(int n):
    """Compute the product of n distinct 2x2 matrices mod M.

    Args:
        n: Number of matrices to multiply.

    Returns:
        Tuple of (result[0][0], result[0][1], result[1][0], result[1][1]) mod M.
    """
    cdef long long MOD = 1000000007
    cdef long long a, b, c, d, na, nb, nc, nd, ma, mb, mc, md
    cdef int i

    # Identity matrix
    a = 1
    b = 0
    c = 0
    d = 1

    with nogil:
        for i in range(n):
            ma = (i * 7 + 1) % 97 + 1
            mb = (i * 5 + 3) % 89 + 1
            mc = (i * 3 + 7) % 79 + 1
            md = (i * 11 + 2) % 83 + 1

            na = (a * ma + b * mc) % MOD
            nb = (a * mb + b * md) % MOD
            nc = (c * ma + d * mc) % MOD
            nd = (c * mb + d * md) % MOD

            a = na
            b = nb
            c = nc
            d = nd

    return (a, b, c, d)
