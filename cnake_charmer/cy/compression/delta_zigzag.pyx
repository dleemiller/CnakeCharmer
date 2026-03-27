# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Delta encoding with zigzag transformation (Cython-optimized).

Keywords: compression, delta, zigzag, encoding, lossless, cython, benchmark
"""

from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(5000000,))
def delta_zigzag(int n):
    """Delta-encode with zigzag transform using typed variables."""
    cdef int i, prev, curr, delta
    cdef long long zigzag_sum, zz, max_zigzag
    cdef int decoded
    cdef long long running

    # Encode phase
    prev = (0 * 13 + 7) % 2000 - 1000
    delta = prev
    if delta >= 0:
        zz = delta << 1
    else:
        zz = ((-delta) << 1) - 1
    zigzag_sum = zz
    max_zigzag = zz

    for i in range(1, n):
        curr = (i * 13 + 7) % 2000 - 1000
        delta = curr - prev
        if delta >= 0:
            zz = delta << 1
        else:
            zz = ((-delta) << 1) - 1
        zigzag_sum += zz
        if zz > max_zigzag:
            max_zigzag = zz
        prev = curr

    # Decode phase
    prev = (0 * 13 + 7) % 2000 - 1000
    delta = prev
    if delta >= 0:
        zz = delta << 1
    else:
        zz = ((-delta) << 1) - 1

    if zz & 1:
        decoded = -(<int>((zz + 1) >> 1))
    else:
        decoded = <int>(zz >> 1)
    running = decoded

    for i in range(1, n):
        curr = (i * 13 + 7) % 2000 - 1000
        delta = curr - prev
        if delta >= 0:
            zz = delta << 1
        else:
            zz = ((-delta) << 1) - 1
        if zz & 1:
            decoded = -(<int>((zz + 1) >> 1))
        else:
            decoded = <int>(zz >> 1)
        running += decoded
        prev = curr

    return (int(zigzag_sum), int(max_zigzag), int(running))
