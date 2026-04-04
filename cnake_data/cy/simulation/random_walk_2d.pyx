# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""2D random walk with n steps using deterministic LCG (Cython-optimized).

Keywords: simulation, random walk, 2d, LCG, cython, benchmark
"""

from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(5000000,))
def random_walk_2d(int n):
    """2D random walk: n steps, 4 directions (N/E/S/W), deterministic LCG.

    Direction for step i: dir = (i * 1664525 + 1013904223) & 3
    """
    cdef int i
    cdef int x = 0
    cdef int y = 0
    cdef int max_x = 0
    cdef int min_y = 0
    cdef int steps_at_origin = 0
    cdef unsigned int direction

    with nogil:
        for i in range(n):
            direction = (<unsigned int>i * 1664525 + 1013904223) & 3
            if direction == 0:
                x += 1
            elif direction == 1:
                y += 1
            elif direction == 2:
                x -= 1
            else:
                y -= 1

            if x > max_x:
                max_x = x
            if y < min_y:
                min_y = y
            if x == 0 and y == 0:
                steps_at_origin += 1

    return (x, y, max_x, min_y, steps_at_origin)
