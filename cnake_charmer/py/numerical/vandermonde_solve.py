"""Vandermonde system solver for polynomial interpolation.

Keywords: numerical, vandermonde, polynomial, interpolation, linear algebra, benchmark
"""

import math

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(250,))
def vandermonde_solve(n: int) -> tuple:
    """Solve a Vandermonde system for polynomial interpolation coefficients.

    Constructs a Vandermonde matrix from nodes x[i] = cos(i * pi / (n - 1))
    (Chebyshev nodes on [-1, 1]) and right-hand side b[i] = sin(x[i] + 0.5).
    Solves Vc = b where V is the Vandermonde matrix, using Gaussian elimination.

    Args:
        n: Number of interpolation nodes.

    Returns:
        Tuple of (sum of coefficients, first coefficient, last coefficient).
    """
    pi = math.pi

    # Generate Chebyshev nodes
    nodes = [0.0] * n
    for i in range(n):
        nodes[i] = math.cos(i * pi / (n - 1)) if n > 1 else 0.0

    # Build Vandermonde matrix V[i][j] = nodes[i]^j and RHS b[i]
    ncols = n + 1
    M = [0.0] * (n * ncols)
    for i in range(n):
        power = 1.0
        for j in range(n):
            M[i * ncols + j] = power
            power *= nodes[i]
        M[i * ncols + n] = math.sin(nodes[i] + 0.5)

    # Gaussian elimination with partial pivoting
    for k in range(n):
        # Find pivot
        max_val = abs(M[k * ncols + k])
        max_row = k
        for i in range(k + 1, n):
            val = abs(M[i * ncols + k])
            if val > max_val:
                max_val = val
                max_row = i

        # Swap rows
        if max_row != k:
            for j in range(ncols):
                tmp = M[k * ncols + j]
                M[k * ncols + j] = M[max_row * ncols + j]
                M[max_row * ncols + j] = tmp

        # Eliminate below
        pivot = M[k * ncols + k]
        if abs(pivot) < 1e-15:
            continue
        for i in range(k + 1, n):
            factor = M[i * ncols + k] / pivot
            for j in range(k, ncols):
                M[i * ncols + j] -= factor * M[k * ncols + j]

    # Back substitution
    c = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = M[i * ncols + n]
        for j in range(i + 1, n):
            s -= M[i * ncols + j] * c[j]
        if abs(M[i * ncols + i]) > 1e-15:
            c[i] = s / M[i * ncols + i]

    # Compute return values
    c_sum = 0.0
    for i in range(n):
        c_sum += c[i]

    return (c_sum, c[0], c[n - 1])
