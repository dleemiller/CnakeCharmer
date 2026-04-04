"""Delta decode a flattened 2D array row-by-row using prefix sums.

Keywords: compression, delta decoding, prefix sum, row-wise, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(5000,))
def delta_decode_rows(n: int) -> tuple:
    """Delta decode h=n rows of width w=32, returning discriminating values.

    Each row is prefix-summed with modular byte arithmetic (mod 256).
    Input: arr[i] = ((i * 7 + 13) % 256).

    Args:
        n: Number of rows (h). Width w is fixed at 32.

    Returns:
        Tuple of (sum_of_all, arr_at_midpoint, arr_at_last).
    """
    w = 32
    total = n * w
    arr = [((i * 7 + 13) % 256) for i in range(total)]

    # Delta decode each row: prefix sum mod 256
    for y in range(n):
        offset = y * w
        for x in range(w - 1):
            pos = offset + x
            arr[pos + 1] = (arr[pos + 1] + arr[pos]) % 256

    s = 0
    for v in arr:
        s += v
    mid = arr[total // 2]
    last = arr[total - 1]

    return (s, mid, last)
