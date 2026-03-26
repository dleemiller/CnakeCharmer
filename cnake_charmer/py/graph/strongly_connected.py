"""Count strongly connected components using Tarjan's algorithm.

Keywords: graph, strongly connected components, Tarjan, SCC, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(100000,))
def strongly_connected(n: int) -> int:
    """Count strongly connected components using Tarjan's algorithm.

    Graph with n nodes. Edges: i->(i*3+1)%n, i->(i*7+2)%n.

    Args:
        n: Number of nodes.

    Returns:
        Tuple of (SCC count, largest SCC size).
    """
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for i in range(n):
        adj[i].append((i * 3 + 1) % n)
        adj[i].append((i * 7 + 2) % n)

    # Tarjan's algorithm (iterative to avoid recursion limit)
    index_counter = [0]
    stack = []
    on_stack = [False] * n
    index = [-1] * n
    lowlink = [-1] * n
    scc_count = 0
    largest_scc_size = 0

    for node in range(n):
        if index[node] != -1:
            continue
        # Iterative DFS
        work_stack = [(node, 0)]
        while work_stack:
            v, ei = work_stack[-1]
            if ei == 0:
                index[v] = index_counter[0]
                lowlink[v] = index_counter[0]
                index_counter[0] += 1
                stack.append(v)
                on_stack[v] = True
            recurse = False
            for idx in range(ei, len(adj[v])):
                w = adj[v][idx]
                if index[w] == -1:
                    work_stack[-1] = (v, idx + 1)
                    work_stack.append((w, 0))
                    recurse = True
                    break
                elif on_stack[w]:
                    if lowlink[v] > lowlink[w]:
                        lowlink[v] = lowlink[w]
            if recurse:
                continue
            work_stack.pop()
            if work_stack:
                parent = work_stack[-1][0]
                if lowlink[parent] > lowlink[v]:
                    lowlink[parent] = lowlink[v]
            if lowlink[v] == index[v]:
                scc_count += 1
                scc_size = 0
                while True:
                    w = stack.pop()
                    on_stack[w] = False
                    scc_size += 1
                    if w == v:
                        break
                if scc_size > largest_scc_size:
                    largest_scc_size = scc_size

    return (scc_count, largest_scc_size)
