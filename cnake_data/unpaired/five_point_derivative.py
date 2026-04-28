def five_point_forward_derivative(data, h):
    """Compute forward finite-difference derivative with 5-point stencil."""
    n = len(data)
    if n < 5:
        return []

    den = 12.0 * h
    deriv = [0.0 for _ in range(n - 4)]

    for i in range(n - 4):
        deriv[i] = (
            -25.0 * data[i]
            + 48.0 * data[i + 1]
            - 36.0 * data[i + 2]
            + 16.0 * data[i + 3]
            - 3.0 * data[i + 4]
        ) / den

    return deriv
