def matrix_mult_elementwise(a, b):
    """Elementwise multiply two equally-shaped complex matrices."""
    n = len(a)
    m = len(a[0]) if n else 0
    out = [[0j for _ in range(m)] for _ in range(n)]

    for i in range(n):
        for j in range(m):
            out[i][j] = a[i][j] * b[i][j]
    return out
