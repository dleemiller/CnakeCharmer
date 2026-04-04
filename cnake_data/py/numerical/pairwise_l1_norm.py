"""Pairwise L1 distance matrix computation.

Given two sets of points (N1xK and N2xK), compute the N1xN2 distance
matrix where each entry is the L1 (Manhattan) distance between point
pairs: sum of |D1[i,k] - D2[j,k]| for all k.

Keywords: pairwise, distance, L1, manhattan, numerical, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(80, 80, 10))
def pairwise_l1_norm(n1: int, n2: int, k: int) -> tuple:
    """Compute pairwise L1 distance matrix between two point sets.

    Points are generated deterministically from indices.

    Args:
        n1: Number of points in first set.
        n2: Number of points in second set.
        k: Dimensionality of each point.

    Returns:
        Tuple of (total_sum, max_distance, distance_at_mid_mid).
    """
    # Generate deterministic point data from indices
    d1 = [[0.0] * k for _ in range(n1)]
    for i in range(n1):
        for d in range(k):
            d1[i][d] = ((i * 31 + d * 7 + 13) % 1000) / 100.0

    d2 = [[0.0] * k for _ in range(n2)]
    for j in range(n2):
        for d in range(k):
            d2[j][d] = ((j * 37 + d * 11 + 17) % 1000) / 100.0

    # Compute pairwise L1 distance matrix
    result = [[0.0] * n2 for _ in range(n1)]
    for i in range(n1):
        for j in range(n2):
            dist = 0.0
            for d in range(k):
                diff = d1[i][d] - d2[j][d]
                if diff < 0.0:
                    diff = -diff
                dist += diff
            result[i][j] = dist

    # Compute summary statistics
    total_sum = 0.0
    max_distance = 0.0
    for i in range(n1):
        for j in range(n2):
            total_sum += result[i][j]
            if result[i][j] > max_distance:
                max_distance = result[i][j]

    mid_i = n1 // 2
    mid_j = n2 // 2
    distance_at_mid_mid = result[mid_i][mid_j]

    return (total_sum, max_distance, distance_at_mid_mid)
