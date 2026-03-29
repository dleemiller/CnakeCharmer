"""Hopcroft-Karp maximum bipartite matching on a deterministic graph.

Keywords: graph, bipartite, matching, hopcroft-karp, bfs, dfs, benchmark
"""

from collections import deque

from cnake_charmer.benchmarks import python_benchmark

_DEGREE = 5
_PAIRS = ((3, 1), (7, 2), (11, 3), (13, 5), (17, 7))


@python_benchmark(args=(2000,))
def hopcroft_karp(n: int) -> tuple:
    """Maximum bipartite matching using Hopcroft-Karp (BFS layering + DFS augmentation).

    Left nodes 0..n-1, right nodes 0..n-1. Each left node u has edges to
    right nodes (u*p + q) % n for (p, q) in [(3,1),(7,2),(11,3),(13,5),(17,7)].

    Args:
        n: Number of nodes on each side.

    Returns:
        Tuple of (matching_size, matched_right_checksum) where checksum is the
        sum of all matched right-node indices.
    """
    INF = n + 1

    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u in range(n):
        for p, q in _PAIRS:
            adj[u].append((u * p + q) % n)

    match_l = [-1] * n   # match_l[u] = right node matched to left u
    match_r = [-1] * n   # match_r[v] = left node matched to right v
    dist = [0] * n

    def bfs() -> bool:
        queue = deque()
        for u in range(n):
            if match_l[u] == -1:
                dist[u] = 0
                queue.append(u)
            else:
                dist[u] = INF
        found = False
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                w = match_r[v]
                if w == -1:
                    found = True
                elif dist[w] == INF:
                    dist[w] = dist[u] + 1
                    queue.append(w)
        return found

    def dfs(u: int) -> bool:
        for v in adj[u]:
            w = match_r[v]
            if w == -1 or (dist[w] == dist[u] + 1 and dfs(w)):
                match_l[u] = v
                match_r[v] = u
                return True
        dist[u] = INF
        return False

    matching = 0
    while bfs():
        for u in range(n):
            if match_l[u] == -1 and dfs(u):
                matching += 1

    checksum = sum(v for v in match_l if v != -1)
    return (matching, checksum)
