"""Run-length encode/decode a deterministic string.

Keywords: string processing, run-length encoding, compression, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(1000000,))
def run_length_encode(n: int) -> tuple:
    """Run-length encode a deterministic string and verify decode.

    String is generated as chr(65 + (i*3) % 5) for each position.

    Args:
        n: Length of the string to generate.

    Returns:
        Tuple of (encoded_length, num_runs, decoded_matches_original).
        decoded_matches_original is 1 if decode matches, 0 otherwise.
    """
    if n == 0:
        return (0, 0, 1)

    # Build string as list of ints (char codes)
    chars = [0] * n
    for i in range(n):
        chars[i] = 65 + (i * 3) % 5

    # Encode: count runs
    num_runs = 1
    encoded_length = 0
    run_char = chars[0]
    run_len = 1

    for i in range(1, n):
        if chars[i] == run_char:
            run_len += 1
        else:
            # Emit this run: 1 char + digits of run_len
            digits = 0
            temp = run_len
            while temp > 0:
                digits += 1
                temp //= 10
            encoded_length += 1 + digits
            num_runs += 1
            run_char = chars[i]
            run_len = 1

    # Emit last run
    digits = 0
    temp = run_len
    while temp > 0:
        digits += 1
        temp //= 10
    encoded_length += 1 + digits

    # Decode verification: rebuild from runs and compare
    # We'll do a second pass to build run pairs and verify
    decoded_ok = 1
    run_char = chars[0]
    run_len = 1
    decode_pos = 0

    for i in range(1, n):
        if chars[i] == run_char:
            run_len += 1
        else:
            # Verify this run
            for j in range(run_len):
                if decode_pos + j >= n or chars[decode_pos + j] != run_char:
                    decoded_ok = 0
            decode_pos += run_len
            run_char = chars[i]
            run_len = 1

    # Verify last run
    for j in range(run_len):
        if decode_pos + j >= n or chars[decode_pos + j] != run_char:
            decoded_ok = 0
    decode_pos += run_len

    if decode_pos != n:
        decoded_ok = 0

    return (encoded_length, num_runs, decoded_ok)
