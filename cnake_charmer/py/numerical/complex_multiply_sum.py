"""Multiply complex numbers and sum the results.

Keywords: complex, multiply, arithmetic, operator overloading, numerical, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(100000,))
def complex_multiply_sum(n: int) -> tuple:
    """Generate n complex number pairs, multiply them, and sum all products.

    Args:
        n: Number of complex multiplications.

    Returns:
        Tuple of (real_sum, imag_sum).
    """
    total_re = 0.0
    total_im = 0.0

    for i in range(n):
        h1 = ((i * 2654435761 + 1) >> 8) & 0xFFFF
        h2 = ((i * 1103515245 + 3) >> 8) & 0xFFFF
        a_re = (h1 % 201 - 100) / 10.0
        a_im = (h2 % 201 - 100) / 10.0

        h3 = ((i * 6364136223846793005 + 7) >> 16) & 0xFFFF
        h4 = ((i * 3935559000370003845 + 11) >> 16) & 0xFFFF
        b_re = (h3 % 201 - 100) / 10.0
        b_im = (h4 % 201 - 100) / 10.0

        # Complex multiply: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        prod_re = a_re * b_re - a_im * b_im
        prod_im = a_re * b_im + a_im * b_re

        # Accumulate
        total_re += prod_re
        total_im += prod_im

    return (total_re, total_im)
