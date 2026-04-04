"""Deterministic simulated annealing for the Travelling Salesman Problem.

Uses a fixed seed LCG for reproducibility. Returns best distance and
route markers.

Keywords: optimization, simulated annealing, TSP, travelling salesman, combinatorial, benchmark
"""

import math

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(500,))
def simulated_annealing_tsp(n: int) -> tuple:
    """Solve TSP for n cities using deterministic simulated annealing.

    Cities placed at deterministic positions. Uses LCG PRNG for reproducibility.

    Args:
        n: Number of cities.

    Returns:
        Tuple of (best_distance, first_city, mid_city).
    """
    # Generate city coordinates deterministically
    cx = [0.0] * n
    cy = [0.0] * n
    for i in range(n):
        cx[i] = math.sin(i * 0.7 + 1.0) * 100.0
        cy[i] = math.cos(i * 0.3 + 2.0) * 100.0

    # Distance function (inline)
    def dist(a, b):
        dx = cx[a] - cx[b]
        dy = cy[a] - cy[b]
        return math.sqrt(dx * dx + dy * dy)

    # Compute tour distance from scratch
    def tour_distance(t):
        d = 0.0
        for i in range(n):
            d += dist(t[i], t[(i + 1) % n])
        return d

    # Initial tour: 0, 1, 2, ..., n-1
    tour = list(range(n))
    current_dist = tour_distance(tour)

    best_tour = tour[:]
    best_distance = current_dist

    # LCG PRNG
    lcg_state = 42
    lcg_a = 1103515245
    lcg_c = 12345
    lcg_m = 2147483648  # 2^31

    temp = 100.0
    cooling = 0.9995
    iterations = n * 200

    for _ in range(iterations):
        # Generate two random indices
        lcg_state = (lcg_a * lcg_state + lcg_c) % lcg_m
        i = lcg_state % n
        lcg_state = (lcg_a * lcg_state + lcg_c) % lcg_m
        j = lcg_state % n

        if i == j:
            continue

        if i > j:
            i, j = j, i

        # Compute delta for 2-opt swap
        i_prev = (i - 1) % n
        j_next = (j + 1) % n

        old_cost = dist(tour[i_prev], tour[i]) + dist(tour[j], tour[j_next])
        new_cost = dist(tour[i_prev], tour[j]) + dist(tour[i], tour[j_next])
        delta = new_cost - old_cost

        if delta < 0.0:
            accept = True
        else:
            lcg_state = (lcg_a * lcg_state + lcg_c) % lcg_m
            rand_val = lcg_state / lcg_m
            accept = bool(temp > 1e-10 and rand_val < math.exp(-delta / temp))

        if accept:
            # Reverse segment [i..j]
            left = i
            right = j
            while left < right:
                tour[left], tour[right] = tour[right], tour[left]
                left += 1
                right -= 1
            current_dist += delta

            if current_dist < best_distance:
                best_distance = current_dist
                best_tour = tour[:]

        temp *= cooling

    # Recompute best_distance from scratch to avoid drift
    best_distance = tour_distance(best_tour)

    return (best_distance, best_tour[0], best_tour[n // 2])
