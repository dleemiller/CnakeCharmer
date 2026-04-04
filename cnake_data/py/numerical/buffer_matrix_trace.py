"""Matrix trace via buffer-backed 2D double array.

Fills an n x n matrix with hash-derived doubles using nested
lists, then computes the trace (sum of diagonal elements).

Keywords: numerical, buffer protocol, matrix, trace, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(200,))
def buffer_matrix_trace(n: int) -> float:
    """Compute trace of a hash-filled n x n matrix.

    Args:
        n: Matrix dimension (n x n).

    Returns:
        Trace of the matrix as a float.
    """
    mask = 0xFFFFFFFF
    matrix = []
    for i in range(n):
        row = [0.0] * n
        for j in range(n):
            idx = i * n + j
            h = ((idx * 2654435761) & mask) ^ ((idx * 2246822519) & mask)
            row[j] = (h & 0xFFFF) / 65535.0
        matrix.append(row)

    trace = 0.0
    for i in range(n):
        trace += matrix[i][i]
    return trace
