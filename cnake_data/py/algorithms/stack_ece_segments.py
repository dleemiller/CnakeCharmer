"""Find ECE-like segments using prefix extrema scanning.

Adapted from The Stack v2 Cython candidate:
- blob_id: ec852b83cf303a1ec1503655572dd9143cbadb35
- filename: find_ECE.pyx

Keywords: algorithms, prefix sums, interval detection, scanning
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(45000, 24))
def stack_ece_segments(length: int, min_len: int) -> tuple:
    """Generate +/- sequence and return ECE interval summary."""
    s = [0] * (length + 1)
    state = 987654321
    for i in range(1, length + 1):
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        s[i] = 1 if (state & 7) < 5 else -4

    L = len(s)
    r = [0] * L
    X = [0] * L
    Y = [0] * L

    for i in range(1, L):
        r[i] = r[i - 1] + s[i]

    X[0] = 0
    for i in range(1, L):
        X[i] = X[i - 1] if X[i - 1] <= r[i] else r[i]

    Y[L - 1] = r[L - 1]
    for i in range(L - 2, -1, -1):
        Y[i] = Y[i + 1] if Y[i + 1] >= r[i] else r[i]

    i = 0
    j = 0
    segs = []
    while j < L:
        if j == L - 1 or Y[j + 1] < X[i]:
            if j - i >= min_len:
                segs.append((i + 1, j))
            j += 1
            while j < L and i < L and Y[j] < X[i]:
                i += 1
        else:
            j += 1

    if segs:
        total_span = sum(b - a + 1 for a, b in segs)
        return (len(segs), segs[0][0], segs[0][1], total_span)
    return (0, 0, 0, 0)
