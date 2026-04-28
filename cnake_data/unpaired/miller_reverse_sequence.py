from fractions import Fraction


def miller_reverse_value(n):
    """Compute sequence value using a reverse-stable recurrence."""
    if n < 1:
        raise ValueError("n must be >= 1")

    y = [0.0 for _ in range(n)]
    a_prev = Fraction(2)
    a_curr = Fraction(-4)

    for k in range(n - 1, -1, -1):
        a_prev, a_curr = a_curr, Fraction(111 - (1130 / a_curr) + (3000 / (a_prev * a_curr)))
        y[k] = float(a_curr)

    return y[0]


def miller_reverse_values(n_values):
    """Compute the reverse-stable value for each n in n_values."""
    return [miller_reverse_value(n) for n in n_values]
