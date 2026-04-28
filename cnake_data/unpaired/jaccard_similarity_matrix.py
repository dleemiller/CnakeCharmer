def jaccard_similarity_matrix(x, y):
    """Compute pairwise Jaccard similarity between binary row vectors.

    Args:
        x: Matrix (list of rows) containing 0/1 values.
        y: Matrix (list of rows) containing 0/1 values.

    Returns:
        Matrix S where S[i][j] is Jaccard similarity between x[i] and y[j].
    """
    n_mx = len(x)
    n_my = len(y)
    n_g = len(x[0]) if n_mx else 0

    s = [[0.0 for _ in range(n_my)] for _ in range(n_mx)]

    for i in range(n_mx):
        for j in range(n_my):
            inter = 0.0
            union = 0.0
            for k in range(n_g):
                a = x[i][k]
                b = y[j][k]
                if a or b:
                    union += 1.0
                    if a and b:
                        inter += 1.0
            s[i][j] = inter / union if union > 0.0 else 0.0

    return s
