"""Compute sum of elements of the covariance matrix for 3 variables.

Variables: x1[i]=sin(i*0.1), x2[i]=cos(i*0.2), x3[i]=sin(i*0.3) with n observations.
Returns the sum of all 9 elements of the 3x3 covariance matrix.

Keywords: statistics, covariance, matrix, multivariate, benchmark
"""

import math

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(500000,))
def covariance_matrix(n: int) -> float:
    """Compute sum of elements of the 3x3 covariance matrix.

    Args:
        n: Number of observations.

    Returns:
        Sum of all 9 covariance matrix elements.
    """
    # Compute means
    sum1 = 0.0
    sum2 = 0.0
    sum3 = 0.0
    for i in range(n):
        sum1 += math.sin(i * 0.1)
        sum2 += math.cos(i * 0.2)
        sum3 += math.sin(i * 0.3)

    mean1 = sum1 / n
    mean2 = sum2 / n
    mean3 = sum3 / n

    # Compute covariance matrix elements (3x3)
    # cov[a][b] = sum((xa[i]-mean_a)*(xb[i]-mean_b)) / (n-1)
    cov = [[0.0] * 3 for _ in range(3)]

    for i in range(n):
        v1 = math.sin(i * 0.1) - mean1
        v2 = math.cos(i * 0.2) - mean2
        v3 = math.sin(i * 0.3) - mean3
        vals = [v1, v2, v3]
        for a in range(3):
            for b in range(3):
                cov[a][b] += vals[a] * vals[b]

    total = 0.0
    for a in range(3):
        for b in range(3):
            total += cov[a][b] / (n - 1)

    return total
