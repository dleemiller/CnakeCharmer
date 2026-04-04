"""Topological sort of a DAG and reachability count from node 0.

Keywords: algorithms, topological sort, DAG, graph, reachability, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(100000,))
def topological_sort(n: int) -> int:
    """Topological sort a DAG with n nodes, count reachable nodes from 0.

    Edges: i -> (i*3+1) % n if target > source.
    Uses Kahn's algorithm (BFS-based topological sort).

    Args:
        n: Number of nodes in the DAG.

    Returns:
        Number of nodes reachable from node 0 in topological order.
    """
    # Build adjacency list and in-degree array
    adj = [[] for _ in range(n)]
    in_degree = [0] * n

    for i in range(n):
        target = (i * 3 + 1) % n
        if target > i:
            adj[i].append(target)
            in_degree[target] += 1

    # Kahn's algorithm
    queue = []
    for i in range(n):
        if in_degree[i] == 0:
            queue.append(i)

    topo_order = []
    head = 0
    while head < len(queue):
        node = queue[head]
        head += 1
        topo_order.append(node)
        for neighbor in adj[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    # Count nodes reachable from node 0 using topo order
    reachable = [False] * n
    reachable[0] = True
    count = 0

    for node in topo_order:
        if reachable[node]:
            count += 1
            for neighbor in adj[node]:
                reachable[neighbor] = True

    return count
