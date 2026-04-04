"""
Batch inversion of 2×2 Jacobian matrices (from FEM code).

Keywords: numerical, jacobian, matrix inversion, fem, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(2000,))
def jacobian_inverse(n: int) -> tuple:
    """Batch-invert 2×2 Jacobian matrices for n elements × 4 points.

    J layout: flat array of size n*m*4 (m=4 points, 4 components per matrix).
    J[pos] = (i * 1000 + j * 100 + k + 1) * 0.01  where pos = m*4*i + 4*j + k

    For each (i,j): [a,b,c,d] -> f=1/(a*d-b*c), invJ=[f*d, -f*b, -f*c, f*a]

    Args:
        n: Number of elements.

    Returns:
        (int(sum_diag * 1e6) % 10**9, int(invJ_at_midpoint_0 * 1e9))
    """
    m = 4
    size = n * m * 4
    J = [0.0] * size
    invJ = [0.0] * size

    for i in range(n):
        for j in range(m):
            base = m * 4 * i + 4 * j
            for k in range(4):
                J[base + k] = (i * 1000 + j * 100 + k + 1) * 0.01

    sum_diag = 0.0
    for i in range(n):
        for j in range(m):
            pos = m * 4 * i + 4 * j
            a = J[pos + 0]
            b = J[pos + 1]
            c = J[pos + 2]
            d = J[pos + 3]
            f = 1.0 / (a * d - b * c)
            invJ[pos + 0] = f * d
            invJ[pos + 1] = -f * b
            invJ[pos + 2] = -f * c
            invJ[pos + 3] = f * a
            sum_diag += invJ[pos + 0]

    mid_pos = m * 4 * (n // 2) + 0
    invJ_mid = invJ[mid_pos]
    return (int(sum_diag * 1e6) % (10**9), int(invJ_mid * 1e9))
