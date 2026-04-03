from math import exp, log


def entropy_score(n):
    """Compute an entropy score from a deterministic distance matrix.

    Builds an n x n distance matrix where dist[i][j] = |i - j| / n,
    then computes a kernel-based entropy estimate:
        E = -(1/n) * sum_j( log( (1/n) * sum_i( exp(-dist[i][j]) ) ) )

    This measures how spread out the points are using a soft nearest-neighbor
    kernel, commonly used in cluster quality assessment.

    Args:
        n: Number of points (matrix dimension).

    Returns:
        (entropy, min_column_kernel, max_column_kernel) rounded to 10 decimals.
    """
    # Build distance matrix
    dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            dist[i][j] = abs(i - j) / n

    # Compute entropy
    s1 = 0.0
    min_col = float("inf")
    max_col = float("-inf")

    for j in range(n):
        s2 = 0.0
        for i in range(n):
            s2 += exp(-dist[i][j])
        s2 = s2 / n
        if s2 < min_col:
            min_col = s2
        if s2 > max_col:
            max_col = s2
        s1 += log(s2)
    s1 = s1 / n

    entropy = -s1

    return (round(entropy, 10), round(min_col, 10), round(max_col, 10))
