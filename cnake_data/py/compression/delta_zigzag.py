"""Delta encoding with zigzag transformation for signed deltas.

Keywords: compression, delta, zigzag, encoding, lossless, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(5000000,))
def delta_zigzag(n: int) -> tuple:
    """Delta-encode with zigzag transform, decode, and verify round-trip.

    Values: v[i] = (i*13 + 7) % 2000 - 1000 (signed range).
    Delta: d[i] = v[i] - v[i-1], d[0] = v[0].
    Zigzag: z = (d << 1) ^ (d >> 31) maps signed to unsigned.
    Decodes back and verifies, returns stats.

    Args:
        n: Number of values.

    Returns:
        Tuple of (zigzag_sum, max_zigzag, decoded_last_value).
    """
    # Encode phase: compute deltas and zigzag
    prev = (0 * 13 + 7) % 2000 - 1000
    first_delta = prev
    # zigzag of first value
    zz = (first_delta << 1) ^ (first_delta >> 31)
    zigzag_sum = zz
    max_zigzag = zz

    for i in range(1, n):
        curr = (i * 13 + 7) % 2000 - 1000
        delta = curr - prev
        # Zigzag encode: map signed to unsigned
        zz = delta << 1 if delta >= 0 else (-delta << 1) - 1
        zigzag_sum += zz
        if zz > max_zigzag:
            max_zigzag = zz
        prev = curr

    # Decode phase: reverse zigzag and accumulate
    prev = (0 * 13 + 7) % 2000 - 1000
    first_delta = prev
    zz = first_delta << 1 if first_delta >= 0 else (-first_delta << 1) - 1

    # Reverse zigzag
    decoded = -(zz + 1 >> 1) if zz & 1 else zz >> 1
    running = decoded

    for i in range(1, n):
        curr = (i * 13 + 7) % 2000 - 1000
        delta = curr - prev
        zz = delta << 1 if delta >= 0 else (-delta << 1) - 1
        # Reverse zigzag
        decoded = -(zz + 1 >> 1) if zz & 1 else zz >> 1
        running += decoded
        prev = curr

    return (zigzag_sum, max_zigzag, running)
