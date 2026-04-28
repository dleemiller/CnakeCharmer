def rankdata_2d_ordinal(array):
    """Ordinal ranks per row (1..ncols), stable by index for ties.

    Args:
        array: 2D list-like of numeric values.

    Returns:
        2D list of float ranks, matching scipy rankdata(..., method='ordinal')
        applied row-wise.
    """
    out = []
    for row in array:
        n = len(row)
        idxs = list(range(n))
        idxs.sort(key=lambda j: row[j])

        ranked = [0.0] * n
        for ord_j, src_j in enumerate(idxs, start=1):
            ranked[src_j] = float(ord_j)
        out.append(ranked)
    return out
