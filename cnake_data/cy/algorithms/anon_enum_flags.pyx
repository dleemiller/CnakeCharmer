# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Process permission checks using bitwise flag operations.

Keywords: algorithms, enum, flags, bitwise, permissions, cython, benchmark
"""

from cnake_data.benchmarks import cython_benchmark

cdef enum:
    FLAG_READ = 1
    FLAG_WRITE = 2
    FLAG_EXEC = 4
    FLAG_ADMIN = 8
    FLAG_OWNER = 16
    FLAG_GROUP = 32
    FLAG_OTHER = 64
    FLAG_STICKY = 128


@cython_benchmark(syntax="cy", args=(100000,))
def anon_enum_flags(int n):
    """Process n permission checks with bitwise flags."""
    cdef int granted = 0
    cdef int i
    cdef unsigned int perm, required

    for i in range(n):
        perm = (
            (<unsigned int>(<long long>i
             * <long long>2654435761)
             ^ (<unsigned int>i >> 2))
            & <unsigned int>0xFF
        )
        required = (
            (<unsigned int>(<long long>i
             * <long long>1664525
             + <long long>1013904223) >> 4)
            & <unsigned int>0xFF
        )

        if perm & FLAG_ADMIN:
            granted += 1
        elif (perm & required) == required:
            granted += 1
        elif ((perm & FLAG_OWNER)
              and (required
                   & (FLAG_READ | FLAG_WRITE))):
            granted += 1

        if perm & FLAG_STICKY:
            perm ^= FLAG_EXEC
        if perm & FLAG_GROUP:
            perm |= FLAG_OTHER

        granted += (perm >> 4) & 1
    return granted
