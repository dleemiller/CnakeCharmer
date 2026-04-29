"""Topological path extraction from weighted DAGs."""

from __future__ import annotations

MIN_SCORE = 1.0e-10


def topological_sort(graph: dict[int, list[int]]) -> list[int]:
    order: list[int] = []
    explored: set[int] = set()

    for v in graph:
        if v in explored:
            continue
        fringe = [v]
        while fringe:
            w = fringe[-1]
            if w in explored:
                fringe.pop()
                continue
            new_nodes = [n for n in graph.get(w, []) if n not in explored]
            if new_nodes:
                fringe.extend(new_nodes)
            else:
                explored.add(w)
                order.append(w)
                fringe.pop()
    order.reverse()
    return order


def find_path(
    nodes: list[int], exprs: list[float], succ: list[list[int]], isource: int, isink: int
):
    n = len(nodes)
    min_exprs = [MIN_SCORE] * n
    prevs = [isink] * n
    min_exprs[isource] = exprs[isource]

    for i in range(n):
        min_expr = min_exprs[i]
        for j in succ[i]:
            new_min_expr = min(min_expr, exprs[j])
            if (prevs[j] == isink) or (new_min_expr > min_exprs[j]):
                min_exprs[j] = new_min_expr
                prevs[j] = i

    expr = min_exprs[isink]
    prev = isink
    ipath = [isink]
    while True:
        prev = prevs[prev]
        ipath.append(prev)
        if prev == isource:
            break
    ipath.reverse()

    for idx, x in enumerate(ipath):
        new_expr = exprs[x] - expr
        exprs[x] = MIN_SCORE if new_expr <= MIN_SCORE else new_expr
        ipath[idx] = nodes[x]
    return tuple(ipath), expr


def find_paths(
    graph: dict[int, list[int]],
    exprs_by_node: dict[int, float],
    path_frac: float = 0.0,
    max_paths: int = 0,
):
    nodes = topological_sort(graph)
    n = len(nodes)
    indexes = {node: i for i, node in enumerate(nodes)}
    exprs = [exprs_by_node[node] for node in nodes]
    succ = [[indexes[x] for x in graph.get(nodes[i], [])] for i in range(n)]
    isource = 0
    isink = n - 1

    if exprs[isource] < MIN_SCORE:
        return []

    path, expr = find_path(nodes, exprs, succ, isource, isink)
    results = [(path, expr)]

    lowest_expr = max(MIN_SCORE, expr * path_frac)
    iterations = 1
    while True:
        if max_paths > 0 and iterations >= max_paths:
            break
        path, expr = find_path(nodes, exprs, succ, isource, isink)
        if expr <= lowest_expr:
            break
        results.append((path, expr))
        iterations += 1
    return results
