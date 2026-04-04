"""Analytic eigenvalue computation for 3x3 symmetric matrices.

Uses the trigonometric method to solve the characteristic cubic polynomial
for matrices of the form [[a,d,0],[d,b,e],[0,e,c]].

Keywords: numerical, eigenvalues, symmetric matrix, trigonometric method, benchmark
"""

import math

from cnake_data.benchmarks import python_benchmark


def _eigenvalues_symmetric_3x3(a, b, c, d, e):
    """Compute eigenvalues of symmetric 3x3 matrix [[a,d,0],[d,b,e],[0,e,c]].

    Uses the trigonometric method for the characteristic cubic.
    Returns (lam1, lam2, lam3) sorted smallest to largest.
    """
    # Trace and related invariants
    p1 = d * d + e * e
    q = (a + b + c) / 3.0  # trace / 3

    # Shifted matrix diagonal: a-q, b-q, c-q
    aq = a - q
    bq = b - q
    cq = c - q

    p2 = aq * aq + bq * bq + cq * cq + 2.0 * p1
    p = math.sqrt(p2 / 6.0)

    if p < 1e-15:
        # Matrix is already diagonal (proportional to identity)
        vals = sorted([a, b, c])
        return vals[0], vals[1], vals[2]

    inv_p = 1.0 / p

    # B = (1/p) * (A - q*I), compute det(B)
    b00 = aq * inv_p
    b11 = bq * inv_p
    b22 = cq * inv_p
    b01 = d * inv_p
    b12 = e * inv_p

    # det of [[b00, b01, 0], [b01, b11, b12], [0, b12, b22]]
    det_b = b00 * (b11 * b22 - b12 * b12) - b01 * (b01 * b22) + 0.0

    # det_b / 2 is the argument for acos, clamp to [-1, 1]
    half_det = det_b * 0.5
    if half_det <= -1.0:
        half_det = -1.0
    elif half_det >= 1.0:
        half_det = 1.0

    phi = math.acos(half_det) / 3.0

    # Eigenvalues
    lam1 = q + 2.0 * p * math.cos(phi)
    lam3 = q + 2.0 * p * math.cos(phi + 2.0 * math.pi / 3.0)
    lam2 = 3.0 * q - lam1 - lam3  # trace identity

    # Sort
    if lam1 > lam2:
        lam1, lam2 = lam2, lam1
    if lam2 > lam3:
        lam2, lam3 = lam3, lam2
    if lam1 > lam2:
        lam1, lam2 = lam2, lam1

    return lam1, lam2, lam3


@python_benchmark(args=(50000,))
def eigenvalues_3x3(n):
    """Compute eigenvalues of n deterministic symmetric 3x3 matrices.

    For each index i, builds a symmetric matrix [[a,d,0],[d,b,e],[0,e,c]]
    with a = i*0.1+1, b = i*0.2+2, c = i*0.15+1.5, d = i*0.05+0.3, e = i*0.03+0.2.
    Computes eigenvalues analytically using the trigonometric method.

    Args:
        n: Number of matrices to diagonalize.

    Returns:
        Tuple of (sum_of_smallest_eigenvalues, sum_of_largest_eigenvalues, trace_check).
    """
    sum_min = 0.0
    sum_max = 0.0
    trace_check = 0.0

    for i in range(n):
        a = i * 0.1 + 1.0
        b = i * 0.2 + 2.0
        c = i * 0.15 + 1.5
        d = i * 0.05 + 0.3
        e = i * 0.03 + 0.2

        lam1, lam2, lam3 = _eigenvalues_symmetric_3x3(a, b, c, d, e)

        sum_min += lam1
        sum_max += lam3
        trace_check += (lam1 + lam2 + lam3) - (a + b + c)

    return (sum_min, sum_max, trace_check)
