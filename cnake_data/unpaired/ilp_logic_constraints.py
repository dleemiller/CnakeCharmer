def sum_vars(xs):
    """Sum variables or signed literals represented as (x, sign)."""
    if not xs:
        return 0
    if isinstance(xs[0], tuple):
        total = 0
        for x, sign in xs:
            if sign == 1:
                total += x
            elif sign == -1:
                total += 1 - x
        return total
    return sum(xs)


def add_and_constraints(xs, z):
    """Return booleans for AND linearization constraints.

    n*z <= sum(xs) <= (n-1) + z
    """
    n = len(xs)
    s = sum_vars(xs)
    return (n * z <= s, s <= (n - 1) + z)


def add_or_constraints(xs, z):
    """Return booleans for OR linearization constraints.

    z <= sum(xs) <= n*z
    """
    n = len(xs)
    s = sum_vars(xs)
    return (z <= s, s <= n * z)
