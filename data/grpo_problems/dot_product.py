import math


def dot_product(n):
    """Compute dot product of two deterministic float vectors of length n.

    Returns (dot_product_value, sum_of_squares_a, sum_of_squares_b).
    """
    a = [0.0] * n
    b = [0.0] * n
    for i in range(n):
        a[i] = math.exp(-i * 0.001)
        b[i] = math.sin(i * 0.01)

    dot = 0.0
    ssa = 0.0
    ssb = 0.0
    for i in range(n):
        dot += a[i] * b[i]
        ssa += a[i] * a[i]
        ssb += b[i] * b[i]
    return (round(dot, 6), round(ssa, 6), round(ssb, 6))
