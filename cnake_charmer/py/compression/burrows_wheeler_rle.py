"""Burrows-Wheeler Transform followed by RLE run counting.

Keywords: compression, burrows-wheeler, bwt, rle, run length, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(2000,))
def burrows_wheeler_rle(n: int) -> int:
    """Compute BWT of a string then count RLE runs.

    Generates string s[i] = chr(65 + (i * 7 + 3) % 4) (alphabet A-D),
    computes the Burrows-Wheeler Transform, then counts the number of
    runs in the transformed string (run-length encoding run count).

    Args:
        n: Length of the input string.

    Returns:
        Number of RLE runs in the BWT output.
    """
    # Generate deterministic string
    s = [chr(65 + (i * 7 + 3) % 4) for i in range(n)]

    # Build all rotations and sort
    indices = list(range(n))
    indices.sort(key=lambda idx: [s[(idx + k) % n] for k in range(n)])

    # Extract last column (BWT output)
    bwt = [s[(idx + n - 1) % n] for idx in indices]

    # Count RLE runs
    if n == 0:
        return 0

    runs = 1
    for i in range(1, n):
        if bwt[i] != bwt[i - 1]:
            runs += 1

    return runs
