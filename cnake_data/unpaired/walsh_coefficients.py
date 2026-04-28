def bit_and_popcount(x, y):
    """Return popcount of bitwise AND for two equal-length bit strings."""
    total = 0
    for a, b in zip(x, y, strict=False):
        if a == "1" and b == "1":
            total += 1
    return total


def walsh_subsum(bitstrings, pivot, values):
    """Compute signed subsum for one Walsh coefficient pivot."""
    s = 0.0
    for idx in range(len(bitstrings)):
        if bit_and_popcount(bitstrings[idx], pivot) % 2 == 0:
            s += values[idx]
        else:
            s -= values[idx]
    return s


def compute_walsh_coefficients(bitstrings, values):
    """Compute Walsh coefficients for all bitstrings in the basis."""
    factor = 1.0 / len(values)
    return [
        factor * walsh_subsum(bitstrings, bitstrings[j], values) for j in range(len(bitstrings))
    ]
