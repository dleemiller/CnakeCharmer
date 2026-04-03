def ks_max_distance(n):
    """Compute the Kolmogorov-Smirnov maximum distance statistic.

    Generates a sorted reference distribution and a weight distribution,
    then walks through both CDFs simultaneously to find the maximum
    absolute difference. This is the core of the KS test for comparing
    two distributions.

    Args:
        n: Number of sample points.

    Returns:
        (max_distance, cdf_reference_final, cdf_weights_final) rounded to 10 decimals.
    """
    # Generate sorted reference values (deterministic, increasing)
    m_sorted = [0.0] * n
    for i in range(n):
        m_sorted[i] = (i * 7 + 3) % (n * 2) + (i * 0.1)
    m_sorted.sort()

    # Generate weight distribution (positive, sums to ~1)
    weights = [0.0] * n
    total_w = 0.0
    for i in range(n):
        weights[i] = ((i * 13 + 7) % 50 + 1) / 100.0
        total_w += weights[i]
    for i in range(n):
        weights[i] /= total_w

    # Compute KS max distance
    m_step = 1.0 / n
    counter_m = 0.0
    counter_c = 0.0
    max_dist = 0.0

    for i in range(n - 1):
        counter_m += m_step
        counter_c += weights[i]

        # Only measure distance at value changes
        if m_sorted[i] != m_sorted[i + 1]:
            distance = counter_m - counter_c
            if distance < 0:
                distance = -distance
            if distance > max_dist:
                max_dist = distance

    return (round(max_dist, 10), round(counter_m, 10), round(counter_c, 10))
