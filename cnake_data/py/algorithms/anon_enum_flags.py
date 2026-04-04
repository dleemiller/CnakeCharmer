"""Process permission checks using bitwise flag operations.

Keywords: algorithms, enum, flags, bitwise, permissions, benchmark
"""

from cnake_data.benchmarks import python_benchmark

FLAG_READ = 1
FLAG_WRITE = 2
FLAG_EXEC = 4
FLAG_ADMIN = 8
FLAG_OWNER = 16
FLAG_GROUP = 32
FLAG_OTHER = 64
FLAG_STICKY = 128


@python_benchmark(args=(100000,))
def anon_enum_flags(n: int) -> int:
    """Process n permission checks using bitwise flag operations.

    Args:
        n: Number of permission checks to process.

    Returns:
        Accumulated count of permission grants.
    """
    granted = 0
    for i in range(n):
        # Build permission mask deterministically
        perm = ((i * 2654435761) ^ (i >> 2)) & 0xFF
        # Build required mask
        required = ((i * 1664525 + 1013904223) >> 4) & 0xFF

        # Check various flag conditions
        if (
            (perm & FLAG_ADMIN)
            or (perm & required) == required
            or ((perm & FLAG_OWNER) and (required & (FLAG_READ | FLAG_WRITE)))
        ):
            granted += 1

        # Toggle flags
        if perm & FLAG_STICKY:
            perm ^= FLAG_EXEC
        if perm & FLAG_GROUP:
            perm |= FLAG_OTHER

        granted += (perm >> 4) & 1
    return granted
