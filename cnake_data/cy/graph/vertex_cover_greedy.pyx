# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Greedy vertex cover approximation (Cython-optimized).

Keywords: graph, vertex cover, greedy, approximation, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from libc.string cimport memset
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(2000,))
def vertex_cover_greedy(int n):
    """Compute a greedy vertex cover using C arrays."""
    if n < 2:
        return (0, 0, 0)

    cdef int i, j, k, u, v

    # First, count unique edges using a temporary structure
    # We'll build edge list, then deduplicate
    # Max possible edges: n (ring) + n/3 (cross)
    cdef int max_edges = n + n / 3 + 1
    cdef int *edge_u = <int *>malloc(max_edges * sizeof(int))
    cdef int *edge_v = <int *>malloc(max_edges * sizeof(int))

    if not edge_u or not edge_v:
        if edge_u: free(edge_u)
        if edge_v: free(edge_v)
        raise MemoryError()

    # Build edge list (canonical: u < v)
    # We need to deduplicate. Use a simple approach: build all, sort, dedup.
    cdef int ne = 0

    # Ring edges
    for i in range(n):
        j = (i + 1) % n
        if i < j:
            edge_u[ne] = i
            edge_v[ne] = j
        else:
            edge_u[ne] = j
            edge_v[ne] = i
        ne += 1

    # Cross edges every 3rd node
    for i in range(0, n, 3):
        k = (i * 7 + 3) % n
        if k != i:
            if i < k:
                u = i
                v = k
            else:
                u = k
                v = i
            # Check if this is a duplicate ring edge
            # Only add if not already a ring neighbor
            if not (v == u + 1 or (u == 0 and v == n - 1)):
                edge_u[ne] = u
                edge_v[ne] = v
                ne += 1

    # Sort edges to deduplicate (simple bubble-free approach: use a hash set via arrays)
    # For correctness, use a mark array approach
    # Actually, we can have duplicate cross-edges too. Let's use a simple sort+dedup.
    # Use insertion sort on (u, v) pairs since ne is manageable
    cdef int tmp_u, tmp_v, pos
    for i in range(1, ne):
        tmp_u = edge_u[i]
        tmp_v = edge_v[i]
        pos = i - 1
        while pos >= 0 and (edge_u[pos] > tmp_u or (edge_u[pos] == tmp_u and edge_v[pos] > tmp_v)):
            edge_u[pos + 1] = edge_u[pos]
            edge_v[pos + 1] = edge_v[pos]
            pos -= 1
        edge_u[pos + 1] = tmp_u
        edge_v[pos + 1] = tmp_v

    # Deduplicate
    cdef int unique_ne = 0
    for i in range(ne):
        if i == 0 or edge_u[i] != edge_u[i - 1] or edge_v[i] != edge_v[i - 1]:
            edge_u[unique_ne] = edge_u[i]
            edge_v[unique_ne] = edge_v[i]
            unique_ne += 1

    cdef int total_edges = unique_ne

    # Build CSR adjacency from unique edges
    cdef int *degree = <int *>malloc(n * sizeof(int))
    cdef int *adj_offset = <int *>malloc((n + 1) * sizeof(int))
    cdef int *adj_list = <int *>malloc(2 * total_edges * sizeof(int))
    cdef int *adj_pos = <int *>malloc(n * sizeof(int))
    cdef int *removed = <int *>malloc(n * sizeof(int))
    cdef int *cur_degree = <int *>malloc(n * sizeof(int))
    # Edge active flags
    cdef int *edge_active = <int *>malloc(total_edges * sizeof(int))
    # For each node, store which edges it belongs to
    # We need neighbor -> edge index mapping. Use adj with edge indices.
    cdef int *adj_edge_idx = <int *>malloc(2 * total_edges * sizeof(int))

    if (not degree or not adj_offset or not adj_list or not adj_pos or
            not removed or not cur_degree or not edge_active or not adj_edge_idx):
        free(edge_u); free(edge_v)
        if degree: free(degree)
        if adj_offset: free(adj_offset)
        if adj_list: free(adj_list)
        if adj_pos: free(adj_pos)
        if removed: free(removed)
        if cur_degree: free(cur_degree)
        if edge_active: free(edge_active)
        if adj_edge_idx: free(adj_edge_idx)
        raise MemoryError()

    memset(degree, 0, n * sizeof(int))
    for i in range(total_edges):
        degree[edge_u[i]] += 1
        degree[edge_v[i]] += 1
        edge_active[i] = 1

    adj_offset[0] = 0
    for i in range(n):
        adj_offset[i + 1] = adj_offset[i] + degree[i]
        cur_degree[i] = degree[i]
    for i in range(n):
        adj_pos[i] = adj_offset[i]

    for i in range(total_edges):
        u = edge_u[i]
        v = edge_v[i]
        adj_list[adj_pos[u]] = v
        adj_edge_idx[adj_pos[u]] = i
        adj_pos[u] += 1
        adj_list[adj_pos[v]] = u
        adj_edge_idx[adj_pos[v]] = i
        adj_pos[v] += 1

    memset(removed, 0, n * sizeof(int))

    cdef int cover_size = 0
    cdef long cover_index_sum = 0
    cdef int remaining_edges = total_edges
    cdef int best, best_deg, ei

    while remaining_edges > 0:
        # Find node with max current degree
        best = -1
        best_deg = -1
        for i in range(n):
            if removed[i] == 0 and cur_degree[i] > best_deg:
                best_deg = cur_degree[i]
                best = i

        if best == -1 or best_deg == 0:
            break

        # Add to cover
        removed[best] = 1
        cover_size += 1
        cover_index_sum += best

        # Remove all active edges incident to best
        for pos in range(adj_offset[best], adj_offset[best + 1]):
            ei = adj_edge_idx[pos]
            if edge_active[ei]:
                edge_active[ei] = 0
                remaining_edges -= 1
                v = adj_list[pos]
                if removed[v] == 0:
                    cur_degree[v] -= 1

        cur_degree[best] = 0

    free(edge_u); free(edge_v)
    free(degree); free(adj_offset); free(adj_list); free(adj_pos)
    free(removed); free(cur_degree); free(edge_active); free(adj_edge_idx)

    return (cover_size, cover_index_sum, total_edges)
