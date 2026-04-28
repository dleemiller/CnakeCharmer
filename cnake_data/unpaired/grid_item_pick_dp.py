def max_value_with_row_pick_limit(r, c, items):
    """Maximize collected value moving right/down with at most 3 picks per row.

    Args:
        r: Number of rows.
        c: Number of columns.
        items: Iterable of (row, col, value) using 0-based coordinates.
    """
    v = [[0 for _ in range(c)] for _ in range(r)]
    for i, j, val in items:
        v[i][j] = val

    # dp[i][j][k]: max value reaching (i,j) having picked k items in row i.
    dp = [[[0 for _ in range(4)] for _ in range(c)] for _ in range(r)]

    for k in range(1, 4):
        dp[0][0][k] = v[0][0]

    for j in range(1, c):
        for k in range(1, 4):
            dp[0][j][k] = max(dp[0][j - 1][k], dp[0][j - 1][k - 1] + v[0][j])

    for i in range(1, r):
        top_best = max(dp[i - 1][0])
        dp[i][0][0] = top_best
        for k in range(1, 4):
            dp[i][0][k] = top_best + v[i][0]

    for i in range(1, r):
        for j in range(1, c):
            up_best = max(dp[i - 1][j])
            left = dp[i][j - 1]

            dp[i][j][0] = max(left[0], up_best)
            dp[i][j][1] = max(left[0] + v[i][j], left[1], up_best + v[i][j])
            dp[i][j][2] = max(left[1] + v[i][j], left[2])
            dp[i][j][3] = max(left[2] + v[i][j], left[3])

    return max(dp[r - 1][c - 1])
