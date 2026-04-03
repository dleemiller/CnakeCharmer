def sum_of_powers(lo, hi):
    """Compute the sum of n^1.1 for n in [lo, hi].

    Returns the sum as a float.
    """
    total = 0.0
    for n in range(lo, hi + 1):
        total += n**1.1
    return total
