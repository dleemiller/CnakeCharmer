import math


def rms_difference(rows, cols):
    """Compute RMS difference between two deterministic matrices.

    Returns the root mean square of element-wise differences.
    """
    a = [[0.0] * cols for _ in range(rows)]
    b = [[0.0] * cols for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            a[i][j] = math.sin(i * 0.1 + j * 0.2)
            b[i][j] = math.cos(i * 0.15 + j * 0.1)

    rms2 = 0.0
    for i in range(rows):
        for j in range(cols):
            diff = a[i][j] - b[i][j]
            rms2 += diff * diff
    return math.sqrt(rms2)
