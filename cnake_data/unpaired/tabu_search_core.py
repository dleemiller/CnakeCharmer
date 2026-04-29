"""Core tabu-search loop for route neighborhood improvement."""

from __future__ import annotations

import bisect
import collections


def granular(base_arcs, reduced_cost_arcs, reduced_cost_vals, max_cost):
    arcs = set(base_arcs)
    idx = bisect.bisect_right(reduced_cost_vals, max_cost)
    arcs.update(reduced_cost_arcs[:idx])
    return arcs


def tabu_search(
    initial_route,
    initial_cost,
    reduced_cost_arcs,
    reduced_cost_vals,
    iterations,
    tenure,
    propose_neighbors,
    score_route,
):
    base_arcs = {(0, node) for node in set(initial_route) if node != 0}
    base_arcs.update({(node, 0) for node in set(initial_route) if node != 0})

    route = list(initial_route)
    best_route = list(initial_route)
    best_cost = float(initial_cost)

    tabu_list = collections.deque(maxlen=tenure)
    max_cost = reduced_cost_vals[0] if reduced_cost_vals else 0.0
    max_reduced = reduced_cost_vals[-1] if reduced_cost_vals else 0.0
    best_count = 0

    for _ in range(iterations):
        if max_cost >= max_reduced:
            break

        arc_filter = granular(base_arcs, reduced_cost_arcs, reduced_cost_vals, max_cost)
        candidates = propose_neighbors(route, arc_filter, tabu_list)
        if not candidates:
            best_count += 1
            if best_count >= 18:
                max_cost = max_cost + abs(max_cost * 0.1)
                best_count = 0
            continue

        move, next_route = candidates[0]
        tabu_list.append(move)

        cost = score_route(next_route)
        if cost < best_cost:
            best_cost = cost
            best_route = list(next_route)
            route = list(next_route)
            best_count = 0
            max_cost = max_cost - abs(max_cost * 0.2)
        else:
            best_count += 1

    return {"route": best_route, "cost": best_cost}
