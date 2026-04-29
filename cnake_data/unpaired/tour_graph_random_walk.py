"""Random-walk tour time simulation on adjacency-matrix graphs."""

from __future__ import annotations

import random


def tour_graph(graph, start=0, start_method="int", max_iter=-1, seed=-1):
    if seed != -1:
        random.seed(seed)
    n = len(graph)
    visited = [0] * n
    visit_dict = {}

    if start_method == "uniform_random":
        start = random.randrange(n)

    v = start
    visited[v] = 1
    counter = 0

    while sum(visited) != n:
        counter += 1
        neighbors = [i for i, w in enumerate(graph[v]) if w > 0]
        v = random.choice(neighbors)
        visited[v] = 1
        visit_dict.setdefault(v, []).append(counter)
        if max_iter != -1 and counter >= max_iter:
            raise StopIteration("reached MAX_ITER")
    return counter, visit_dict


def run_simulation(graph, n=1000, start=0, start_method="int", max_iter=-1):
    data = [0] * n
    for i in range(n):
        data[i] = tour_graph(graph, start=start, start_method=start_method, max_iter=max_iter)[0]
    return data
