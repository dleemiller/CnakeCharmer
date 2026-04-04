"""Power iteration for dominant eigenvalue.

Keywords: numerical, eigenvalue, power iteration, linear algebra, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(300,))
def power_iteration(n: int) -> float:
    """Find dominant eigenvalue of an n x n matrix using power iteration.

    Constructs matrix M[i][j] = (i * j + 3) % 10 / 10.0, then runs
    100 iterations of the power method to find the largest eigenvalue.

    Args:
        n: Matrix dimension.

    Returns:
        The dominant eigenvalue estimate.
    """
    # Build matrix as flat array
    M = [0.0] * (n * n)
    for i in range(n):
        for j in range(n):
            M[i * n + j] = ((i * j + 3) % 10) / 10.0

    # Initial vector
    v = [1.0] * n

    eigenvalue = 0.0

    for _ in range(100):
        # Matrix-vector multiply
        w = [0.0] * n
        for i in range(n):
            s = 0.0
            for j in range(n):
                s += M[i * n + j] * v[j]
            w[i] = s

        # Find max absolute value for normalization
        max_val = 0.0
        for i in range(n):
            if abs(w[i]) > max_val:
                max_val = abs(w[i])

        if max_val == 0.0:
            break

        eigenvalue = max_val

        # Normalize
        for i in range(n):
            v[i] = w[i] / max_val

    return eigenvalue
