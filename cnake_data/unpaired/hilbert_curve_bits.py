def hil_xy_from_s(s, n):
    """Map Hilbert index ``s`` to (x, y) for order ``n``."""
    if not (0 <= n < 64):
        raise ValueError("n must be in [0, 63]")
    if s >= 4**n:
        raise ValueError("s out of range for n")

    x = 0
    y = 0
    for i in range(0, 2 * n, 2):
        sa = (s >> (i + 1)) & 1
        sb = (s >> i) & 1

        swap = (sa ^ sb) - 1
        cmpl = -(sa & sb)
        x ^= y
        y ^= (x & swap) ^ cmpl
        x ^= y

        x = (x >> 1) | (sa << 63)
        y = (y >> 1) | ((sa ^ sb) << 63)

    return (x >> (64 - n), y >> (64 - n))


def hil_s_from_xy(x, y, n):
    """Map Hilbert coordinate (x, y) to index ``s`` for order ``n``."""
    if not (0 <= n < 64):
        raise ValueError("n must be in [0, 63]")

    s = 0
    for i in range(n - 1, -1, -1):
        xi = (x >> i) & 1
        yi = (y >> i) & 1
        s = 4 * s + 2 * xi + (xi ^ yi)

        x ^= y
        y ^= x & (yi - 1)
        x ^= y

        x ^= -xi & (yi - 1)
        y ^= -xi & (yi - 1)

    return s
