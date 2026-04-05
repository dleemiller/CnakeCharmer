"""Apply a fixed 1D banded Toeplitz operator and summarize output energy.

Keywords: numerical, toeplitz, banded matrix, stencil, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(120000,))
def toeplitz_band_energy(n: int) -> tuple:
    """Compute a 5-point stencil transform and return energy statistics."""
    if n <= 0:
        return (0, 0, 0)

    a0, a1, a2 = 5, -3, 1
    vals = [0] * n
    for i in range(n):
        vals[i] = ((i * 19 + 23) % 211) - 105

    total = 0
    energy = 0
    mid = 0

    for i in range(n):
        y = a0 * vals[i]
        if i > 0:
            y += a1 * vals[i - 1]
        if i + 1 < n:
            y += a1 * vals[i + 1]
        if i > 1:
            y += a2 * vals[i - 2]
        if i + 2 < n:
            y += a2 * vals[i + 2]
        total += y
        energy += y * y
        if i == n // 2:
            mid = y

    return (total, energy % 1_000_000_007, mid)
