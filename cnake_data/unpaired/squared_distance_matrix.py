def squared_distance_matrix(n):
    """Compute a pairwise squared Euclidean distance matrix for n points.

    Generates n points in 3D space using a deterministic formula,
    then computes the full n x n squared distance matrix element by element.
    Returns summary statistics of the distance matrix.

    Args:
        n: Number of points.

    Returns:
        (max_dist, min_nonzero_dist, trace, checksum) rounded to 8 decimals.
    """
    dim = 3

    # Generate deterministic points
    points = [[0.0] * dim for _ in range(n)]
    for i in range(n):
        for d in range(dim):
            points[i][d] = ((i * 17 + d * 31 + 5) % 200) / 50.0 - 2.0

    # Compute squared distance matrix
    dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            d2 = 0.0
            for d in range(dim):
                diff = points[i][d] - points[j][d]
                d2 += diff * diff
            dist[i][j] = d2

    # Compute statistics
    max_dist = 0.0
    min_nonzero = float("inf")
    trace = 0.0
    checksum = 0.0

    for i in range(n):
        for j in range(n):
            if dist[i][j] > max_dist:
                max_dist = dist[i][j]
            if dist[i][j] > 0.0 and dist[i][j] < min_nonzero:
                min_nonzero = dist[i][j]
            if i == j:
                trace += dist[i][j]
            checksum += dist[i][j]

    if min_nonzero == float("inf"):
        min_nonzero = 0.0

    return (round(max_dist, 8), round(min_nonzero, 8), round(trace, 8), round(checksum, 8))
