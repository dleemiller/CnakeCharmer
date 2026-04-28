def helper_prefix_max_alpha(pv_orig, m1):
    """Compute prefix maxima and shifted alpha arrays from p-values."""
    pvm = [0.0 for _ in range(m1)]
    alpha = [0.0 for _ in range(m1 + 1)]
    alpha2 = [0.0 for _ in range(m1)]

    alpha[0] = 0.0
    running = float("-inf")
    for i in range(m1):
        if pv_orig[i] > running:
            running = pv_orig[i]
        pvm[i] = running

    for i in range(1, m1 + 1):
        alpha[i] = pvm[i - 1]
        alpha2[i - 1] = pvm[i - 1] - 0.0000001

    return pvm, alpha, alpha2
