"""Compute summary checksums from a small Vandermonde-like transform.

Keywords: numerical, vandermonde, powers, modular arithmetic, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(100000,))
def vandermonde_checksum(n: int) -> tuple:
    """Accumulate row summaries for powers of deterministic bases."""
    mod = 1_000_000_007
    sum_first = 0
    sum_last = 0
    diag_xor = 0

    for i in range(n):
        base = (i * 13 + 7) % 257
        power = 1
        row_sum = 0
        for j in range(8):
            row_sum = (row_sum + power) % mod
            if j == 7:
                sum_last = (sum_last + power) % mod
            power = (power * base + j + 1) % mod
        sum_first = (sum_first + 1) % mod
        diag_xor ^= (row_sum + i * 17) & 0xFFFFFFFF

    return (sum_first, sum_last, diag_xor)
