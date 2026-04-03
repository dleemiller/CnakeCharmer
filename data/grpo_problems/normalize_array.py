def normalize_array(n):
    """Normalize a deterministic array to [0, 1] range.

    Returns (min_normalized, max_normalized, sum_normalized).
    """
    arr = [0.0] * n
    for i in range(n):
        arr[i] = (i * 37 + 13) % 1000 * 0.001

    amin = arr[0]
    amax = arr[0]
    for v in arr:
        if v < amin:
            amin = v
        if v > amax:
            amax = v

    rng = amax - amin
    if rng == 0:
        rng = 1.0

    total = 0.0
    for i in range(n):
        arr[i] = (arr[i] - amin) / rng
        total += arr[i]

    return (round(arr[0], 6), round(arr[-1], 6), round(total, 4))
