def run_length_encode(n):
    """Run-length encode a deterministic integer sequence.

    Returns (number of runs, total encoded length, checksum of run lengths).
    """
    data = [0] * n
    for i in range(n):
        data[i] = (i * 7 + 13) % 20

    if n == 0:
        return (0, 0, 0)

    num_runs = 1
    total_length = 0
    checksum = 0
    current_val = data[0]
    current_len = 1

    for i in range(1, n):
        if data[i] == current_val:
            current_len += 1
        else:
            total_length += current_len
            checksum = (checksum + current_len * (current_val + 1)) & 0xFFFFFFFF
            num_runs += 1
            current_val = data[i]
            current_len = 1

    total_length += current_len
    checksum = (checksum + current_len * (current_val + 1)) & 0xFFFFFFFF

    return (num_runs, total_length, checksum)
