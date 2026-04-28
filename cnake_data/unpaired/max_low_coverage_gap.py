def find_max_low_coverage_gap(arr, covfactor):
    """Largest contiguous span where arr[i] <= max(arr) * covfactor."""
    if not arr:
        return 0

    emax = max(arr)
    th = emax * covfactor
    emin = min(arr)

    if emin > th:
        return 0

    idx = [i for i, v in enumerate(arr) if v <= th]
    if not idx:
        return 0

    cmax = 1
    cst = idx[0]
    ced = cst

    for i in idx[1:]:
        if i == ced + 1:
            ced = i
        else:
            cmax = max(cmax, ced - cst + 1)
            cst = i
            ced = i

    cmax = max(cmax, ced - cst + 1)
    return cmax
