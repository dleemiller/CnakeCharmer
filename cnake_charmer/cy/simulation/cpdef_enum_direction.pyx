# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Random walk using cpdef enum directions (Cython-optimized).

Keywords: simulation, random walk, cpdef enum, direction, cython, benchmark
"""

from cnake_charmer.benchmarks import cython_benchmark

cpdef enum Direction:
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3


@cython_benchmark(syntax="cy", args=(100000,))
def cpdef_enum_direction(int n):
    """Simulate a random walk using cpdef enum directions."""
    cdef long long x = 0
    cdef long long y = 0
    cdef int i
    cdef int direction
    cdef unsigned long long hash_val

    for i in range(n):
        hash_val = <unsigned long long>i * <unsigned long long>2654435761
        direction = (hash_val >> 16) % 4
        if direction == NORTH:
            y += 1
        elif direction == EAST:
            x += 1
        elif direction == SOUTH:
            y -= 1
        else:
            x -= 1

    cdef long long ax = x if x >= 0 else -x
    cdef long long ay = y if y >= 0 else -y
    return <int>(ax + ay)
