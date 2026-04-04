"""Greedy graph coloring on a deterministic sparse graph.

Keywords: graph coloring, greedy, chromatic, adjacency, algorithm, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(100000,))
def graph_coloring_greedy(n: int) -> int:
    """Greedy graph coloring and return number of colors used.

    Build a graph on n nodes with edges: i->(i*3+1)%n, i->(i*7+2)%n,
    i->(i*11+3)%n (no self-loops). Color nodes greedily in order 0..n-1,
    assigning the smallest color not used by neighbors.

    Args:
        n: Number of nodes.

    Returns:
        Tuple of (number of distinct colors used, color checksum).
    """
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for i in range(n):
        for target in ((i * 3 + 1) % n, (i * 7 + 2) % n, (i * 11 + 3) % n):
            if target != i:
                adj[i].append(target)
                adj[target].append(i)

    colors = [-1] * n
    max_color = 0

    for node in range(n):
        # Find colors used by neighbors
        used = set()
        for neighbor in adj[node]:
            if colors[neighbor] != -1:
                used.add(colors[neighbor])

        # Find smallest available color
        color = 0
        while color in used:
            color += 1
        colors[node] = color
        if color > max_color:
            max_color = color

    num_colors = max_color + 1
    color_checksum = sum(colors[i] * ((i * 31 + 7) % 1000) for i in range(n)) % (10**9 + 7)
    return (num_colors, color_checksum)
