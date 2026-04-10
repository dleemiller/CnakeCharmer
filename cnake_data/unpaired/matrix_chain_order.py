def matrix_chain_order(n):
    """Find minimum scalar multiplications for matrix chain product via O(n^3) DP.

    Returns (min_cost, number of DP cells filled, trace checksum).
    """
    dims = [0] * (n + 1)
    for i in range(n + 1):
        dims[i] = 10 + (i * 37 + 13) % 90

    m = [[0] * n for _ in range(n)]
    cells = 0

    for chain_len in range(2, n + 1):
        for i in range(n - chain_len + 1):
            j = i + chain_len - 1
            m[i][j] = 1 << 62
            for k in range(i, j):
                cost = m[i][k] + m[k + 1][j] + dims[i] * dims[k + 1] * dims[j + 1]
                if cost < m[i][j]:
                    m[i][j] = cost
                cells += 1

    checksum = 0
    for i in range(n):
        for j in range(i, min(i + 5, n)):
            checksum = (checksum * 31 + m[i][j]) & 0xFFFFFFFF

    return (m[0][n - 1], cells, checksum)
