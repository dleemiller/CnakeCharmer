"""Run-length encoding of a byte sequence.

Keywords: compression, RLE, run-length encoding, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(500000,))
def rle_encode(n: int) -> tuple:
    """Run-length encode a byte sequence and return summary statistics.

    Input: byte[i] = (i // 7) & 0xFF — produces runs of length 7.
    RLE: scan left to right, count consecutive identical bytes, emit (count, byte)
    pairs. Maximum run length per pair is 255.

    Args:
        n: Number of input bytes.

    Returns:
        Tuple of (num_runs, total_encoded_bytes, checksum) where
        checksum = sum of all count values % 10**9.
    """
    if n == 0:
        return (0, 0, 0)

    num_runs = 0
    checksum = 0

    current = (0 // 7) & 0xFF
    run_len = 1

    for i in range(1, n):
        b = (i // 7) & 0xFF
        if b == current and run_len < 255:
            run_len += 1
        else:
            num_runs += 1
            checksum += run_len
            current = b
            run_len = 1

    # Final run
    num_runs += 1
    checksum += run_len

    total_encoded_bytes = 2 * num_runs
    return (num_runs, total_encoded_bytes, checksum % (10**9))
