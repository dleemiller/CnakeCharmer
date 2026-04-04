"""Product of n distinct 2x2 integer matrices mod M.

Keywords: matrix, product, modular arithmetic, numerical, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark

MOD = 10**9 + 7


@python_benchmark(args=(500000,))
def matrix_product(n: int) -> tuple:
    """Compute the product of n distinct 2x2 matrices mod M.

    Matrix i has entries:
      a = (i*7+1) % 97 + 1
      b = (i*5+3) % 89 + 1
      c = (i*3+7) % 79 + 1
      d = (i*11+2) % 83 + 1

    Args:
        n: Number of matrices to multiply.

    Returns:
        Tuple of (result[0][0], result[0][1], result[1][0], result[1][1]) mod M.
    """
    # Start with identity matrix
    a, b, c, d = 1, 0, 0, 1

    for i in range(n):
        ma = (i * 7 + 1) % 97 + 1
        mb = (i * 5 + 3) % 89 + 1
        mc = (i * 3 + 7) % 79 + 1
        md = (i * 11 + 2) % 83 + 1

        na = (a * ma + b * mc) % MOD
        nb = (a * mb + b * md) % MOD
        nc = (c * ma + d * mc) % MOD
        nd = (c * mb + d * md) % MOD

        a, b, c, d = na, nb, nc, nd

    return (a, b, c, d)
