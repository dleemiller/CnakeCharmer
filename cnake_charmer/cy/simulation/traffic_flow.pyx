# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Nagel-Schreckenberg traffic flow model, deterministic variant (Cython-optimized).

Keywords: simulation, traffic, Nagel-Schreckenberg, cellular automaton, flow, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(5000,))
def traffic_flow(int n):
    """Simulate deterministic Nagel-Schreckenberg traffic flow on a ring road."""
    cdef int v_max = 5
    cdef int steps = 500
    cdef int n_cars = n // 3
    cdef int i, step, next_car, gap
    cdef long total_flow = 0
    cdef int speed_sum, num_stopped, avg_speed_x1000

    cdef int *positions = <int *>malloc(n_cars * sizeof(int))
    cdef int *speeds = <int *>malloc(n_cars * sizeof(int))
    if not positions or not speeds:
        if positions: free(positions)
        if speeds: free(speeds)
        raise MemoryError()

    for i in range(n_cars):
        positions[i] = i * 3
        speeds[i] = i % (v_max + 1)

    for step in range(steps):
        # Step 1: Acceleration
        for i in range(n_cars):
            if speeds[i] < v_max:
                speeds[i] += 1

        # Step 2: Slowing
        for i in range(n_cars):
            next_car = (i + 1) % n_cars
            gap = positions[next_car] - positions[i]
            if gap <= 0:
                gap += n
            if speeds[i] >= gap:
                speeds[i] = gap - 1
                if speeds[i] < 0:
                    speeds[i] = 0

        # Step 3: Deterministic braking
        for i in range(n_cars):
            if (positions[i] + step) % 7 == 0:
                if speeds[i] > 0:
                    speeds[i] -= 1

        # Step 4: Move
        for i in range(n_cars):
            positions[i] = (positions[i] + speeds[i]) % n
            total_flow += speeds[i]

    # Final statistics
    speed_sum = 0
    num_stopped = 0
    for i in range(n_cars):
        speed_sum += speeds[i]
        if speeds[i] == 0:
            num_stopped += 1

    if n_cars > 0:
        avg_speed_x1000 = speed_sum * 1000 // n_cars
    else:
        avg_speed_x1000 = 0

    free(positions)
    free(speeds)
    return (total_flow, avg_speed_x1000, num_stopped)
