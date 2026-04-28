def segment_average(data, m=0, n=None):
    """Average over data[m:n] following trace semantics."""
    if n is None:
        n = len(data)

    mean = 0.0
    for i in range(m, n):
        mean += data[i]

    return mean / (n - m + 1)
