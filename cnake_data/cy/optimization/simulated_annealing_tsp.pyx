# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Deterministic simulated annealing for TSP (Cython-optimized).

Uses a fixed seed LCG for reproducibility. Returns best distance and
route markers.

Keywords: optimization, simulated annealing, TSP, travelling salesman, combinatorial, cython, benchmark
"""

from libc.math cimport sin, cos, sqrt, exp
from libc.stdlib cimport malloc, free
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(500,))
def simulated_annealing_tsp(int n):
    """Solve TSP for n cities using deterministic simulated annealing."""
    cdef int i, j, k, left, right, i_prev, j_next, iterations
    cdef double current_dist, best_distance, delta, old_cost, new_cost
    cdef double temp, cooling, rand_val
    cdef long long lcg_state, lcg_a, lcg_c, lcg_m
    cdef int tmp
    cdef double dx, dy

    cdef double *cx = <double *>malloc(n * sizeof(double))
    cdef double *cy_arr = <double *>malloc(n * sizeof(double))
    cdef int *tour = <int *>malloc(n * sizeof(int))
    cdef int *best_tour = <int *>malloc(n * sizeof(int))
    if not cx or not cy_arr or not tour or not best_tour:
        raise MemoryError()

    # Generate city coordinates deterministically
    for i in range(n):
        cx[i] = sin(i * 0.7 + 1.0) * 100.0
        cy_arr[i] = cos(i * 0.3 + 2.0) * 100.0

    # Initial tour: 0, 1, 2, ..., n-1
    for i in range(n):
        tour[i] = i

    # Compute initial distance
    current_dist = 0.0
    for i in range(n):
        j = (i + 1) % n
        dx = cx[tour[i]] - cx[tour[j]]
        dy = cy_arr[tour[i]] - cy_arr[tour[j]]
        current_dist += sqrt(dx * dx + dy * dy)

    for i in range(n):
        best_tour[i] = tour[i]
    best_distance = current_dist

    # LCG PRNG
    lcg_state = 42
    lcg_a = 1103515245
    lcg_c = 12345
    lcg_m = 2147483648  # 2^31

    temp = 100.0
    cooling = 0.9995
    iterations = n * 200

    for k in range(iterations):
        # Generate two random indices
        lcg_state = (lcg_a * lcg_state + lcg_c) % lcg_m
        i = lcg_state % n
        lcg_state = (lcg_a * lcg_state + lcg_c) % lcg_m
        j = lcg_state % n

        if i == j:
            continue

        if i > j:
            tmp = i
            i = j
            j = tmp

        # Compute delta for 2-opt swap
        i_prev = (i - 1 + n) % n
        j_next = (j + 1) % n

        dx = cx[tour[i_prev]] - cx[tour[i]]
        dy = cy_arr[tour[i_prev]] - cy_arr[tour[i]]
        old_cost = sqrt(dx * dx + dy * dy)
        dx = cx[tour[j]] - cx[tour[j_next]]
        dy = cy_arr[tour[j]] - cy_arr[tour[j_next]]
        old_cost += sqrt(dx * dx + dy * dy)

        dx = cx[tour[i_prev]] - cx[tour[j]]
        dy = cy_arr[tour[i_prev]] - cy_arr[tour[j]]
        new_cost = sqrt(dx * dx + dy * dy)
        dx = cx[tour[i]] - cx[tour[j_next]]
        dy = cy_arr[tour[i]] - cy_arr[tour[j_next]]
        new_cost += sqrt(dx * dx + dy * dy)

        delta = new_cost - old_cost

        if delta < 0.0:
            # Accept
            left = i
            right = j
            while left < right:
                tmp = tour[left]
                tour[left] = tour[right]
                tour[right] = tmp
                left += 1
                right -= 1
            current_dist += delta
            if current_dist < best_distance:
                best_distance = current_dist
                for i in range(n):
                    best_tour[i] = tour[i]
        else:
            lcg_state = (lcg_a * lcg_state + lcg_c) % lcg_m
            rand_val = <double>lcg_state / <double>lcg_m
            if temp > 1e-10 and rand_val < exp(-delta / temp):
                left = i
                right = j
                while left < right:
                    tmp = tour[left]
                    tour[left] = tour[right]
                    tour[right] = tmp
                    left += 1
                    right -= 1
                current_dist += delta
                if current_dist < best_distance:
                    best_distance = current_dist
                    for i in range(n):
                        best_tour[i] = tour[i]

        temp *= cooling

    # Recompute best_distance from scratch to avoid drift
    best_distance = 0.0
    for i in range(n):
        j = (i + 1) % n
        dx = cx[best_tour[i]] - cx[best_tour[j]]
        dy = cy_arr[best_tour[i]] - cy_arr[best_tour[j]]
        best_distance += sqrt(dx * dx + dy * dy)

    cdef int first_city = best_tour[0]
    cdef int mid_city = best_tour[n // 2]

    free(cx)
    free(cy_arr)
    free(tour)
    free(best_tour)
    return (best_distance, first_city, mid_city)
