"""Count edges traversed in an Eulerian path on a directed graph.

Keywords: graph, euler, eulerian path, directed, traversal, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(100000,))
def euler_path(n: int) -> int:
    """Count edges traversed in an Eulerian path using Hierholzer's algorithm.

    Builds a directed graph with n nodes. Each node i has edges:
      i -> (i+1)%n
      i -> (i+2)%n

    Every node has in-degree 2 and out-degree 2, so an Eulerian circuit exists.
    Returns the number of edges traversed.

    Args:
        n: Number of nodes in the graph.

    Returns:
        Number of edges traversed in the Eulerian circuit.
    """
    if n < 3:
        return 0

    # Build adjacency list as mutable lists
    adj = [[] for _ in range(n)]
    for i in range(n):
        adj[i].append((i + 1) % n)
        adj[i].append((i + 2) % n)

    # Hierholzer's algorithm
    stack = [0]
    path = []
    edge_idx = [0] * n

    while stack:
        v = stack[-1]
        if edge_idx[v] < len(adj[v]):
            u = adj[v][edge_idx[v]]
            edge_idx[v] += 1
            stack.append(u)
        else:
            path.append(stack.pop())

    # Number of edges = number of nodes in path - 1
    return len(path) - 1
