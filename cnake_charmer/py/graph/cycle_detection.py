"""
Count back edges (cycles) via DFS in a directed graph.

Keywords: graph, cycle, dfs, detection, back edge, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(100000,))
def cycle_detection(n: int) -> int:
    """Count the number of back edges found via DFS in a directed graph.

    Builds a deterministic directed graph with n nodes. Each node i has edges:
      i -> (i*3+1)%n
      i -> (i*7+2)%n

    Performs iterative DFS from every unvisited node and counts back edges
    (edges to a node currently on the recursion stack), which indicate cycles.

    Args:
        n: Number of nodes in the graph.

    Returns:
        Number of back edges found during DFS.
    """
    # Build adjacency list
    adj_0 = [(i * 3 + 1) % n for i in range(n)]
    adj_1 = [(i * 7 + 2) % n for i in range(n)]

    WHITE, GRAY, BLACK = 0, 1, 2
    color = [WHITE] * n
    back_edges = 0

    for start in range(n):
        if color[start] != WHITE:
            continue
        # Iterative DFS using explicit stack
        # Stack entries: (node, edge_index)
        stack = [(start, 0)]
        color[start] = GRAY
        while stack:
            u, ei = stack[-1]
            if ei < 2:
                stack[-1] = (u, ei + 1)
                v = adj_0[u] if ei == 0 else adj_1[u]
                if color[v] == WHITE:
                    color[v] = GRAY
                    stack.append((v, 0))
                elif color[v] == GRAY:
                    back_edges += 1
            else:
                color[u] = BLACK
                stack.pop()

    return back_edges
