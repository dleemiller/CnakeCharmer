def bisection_root(n_iters):
    """Find root of f(x) = x^3 - x - 2 using bisection method.

    Runs n_iters iterations starting from [1, 2].
    Returns (root_approximation, interval_width, f_at_root).
    """
    a = 1.0
    b = 2.0

    for _ in range(n_iters):
        mid = (a + b) / 2.0
        f_mid = mid**3 - mid - 2.0
        f_a = a**3 - a - 2.0
        if f_a * f_mid < 0:
            b = mid
        else:
            a = mid

    root = (a + b) / 2.0
    width = b - a
    f_root = root**3 - root - 2.0
    return (round(root, 10), round(width, 10), round(f_root, 10))
