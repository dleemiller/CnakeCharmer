"""Tarjan's strongly connected components algorithm on a deterministic graph.

Keywords: algorithms, graph, tarjan, scc, strongly connected, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(300000,))
def tarjan_scc(n: int) -> tuple:
    """Find SCCs in a deterministic directed graph with n nodes.

    Graph edges: for each node i, add edges to (i*3+7) % n and (i*5+11) % n.

    Args:
        n: Number of nodes in the graph.

    Returns:
        Tuple of (num_components, largest_component_size, smallest_component_size).
    """
    # Build adjacency list
    adj = [None] * n
    for i in range(n):
        e1 = (i * 3 + 7) % n
        e2 = (i * 5 + 11) % n
        if e1 == e2:
            adj[i] = [e1]
        else:
            adj[i] = [e1, e2]

    # Tarjan's algorithm (iterative to avoid stack overflow)
    index_counter = [0]
    indices = [-1] * n
    lowlinks = [-1] * n
    on_stack = [False] * n
    stack = []
    sccs = []

    def strongconnect(v):
        # Iterative version using explicit call stack
        call_stack = [(v, 0)]  # (node, neighbor_index)
        indices[v] = index_counter[0]
        lowlinks[v] = index_counter[0]
        index_counter[0] += 1
        stack.append(v)
        on_stack[v] = True

        while call_stack:
            node, ni = call_stack[-1]
            neighbors = adj[node]

            if ni < len(neighbors):
                call_stack[-1] = (node, ni + 1)
                w = neighbors[ni]
                if indices[w] == -1:
                    indices[w] = index_counter[0]
                    lowlinks[w] = index_counter[0]
                    index_counter[0] += 1
                    stack.append(w)
                    on_stack[w] = True
                    call_stack.append((w, 0))
                elif on_stack[w]:
                    if lowlinks[node] > indices[w]:
                        lowlinks[node] = indices[w]
            else:
                # All neighbors processed
                if lowlinks[node] == indices[node]:
                    scc = []
                    while True:
                        w = stack.pop()
                        on_stack[w] = False
                        scc.append(w)
                        if w == node:
                            break
                    sccs.append(len(scc))

                call_stack.pop()
                if call_stack:
                    parent = call_stack[-1][0]
                    if lowlinks[parent] > lowlinks[node]:
                        lowlinks[parent] = lowlinks[node]

    for i in range(n):
        if indices[i] == -1:
            strongconnect(i)

    num_components = len(sccs)
    largest = 0
    smallest = n + 1
    for s in sccs:
        if s > largest:
            largest = s
        if s < smallest:
            smallest = s

    return (num_components, largest, smallest)
