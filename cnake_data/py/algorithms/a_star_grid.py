"""A* pathfinding on an n x n grid with deterministic obstacles.

Keywords: algorithms, pathfinding, a-star, grid, search, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(300,))
def a_star_grid(n: int) -> tuple:
    """Find shortest path on n x n grid from top-left to bottom-right.

    Obstacles at cells where (i*7 + j*13) % 10 == 0 (except start/end).
    Uses Manhattan distance heuristic.

    Args:
        n: Grid dimension.

    Returns:
        Tuple of (path_length, nodes_explored, path_midpoint_x).
        path_length is -1 if no path found.
    """
    # Obstacle map
    blocked = [False] * (n * n)
    for i in range(n):
        for j in range(n):
            if (i * 7 + j * 13) % 10 == 0:
                blocked[i * n + j] = True
    # Ensure start and end are clear
    blocked[0] = False
    blocked[(n - 1) * n + (n - 1)] = False

    # A* with binary heap (manual min-heap using list)
    INF = n * n + 1
    g_score = [INF] * (n * n)
    g_score[0] = 0
    came_from = [-1] * (n * n)

    # Heap entries: (f_score, node_index)
    heap = [(n - 1 + n - 1, 0)]
    in_closed = [False] * (n * n)
    nodes_explored = 0
    goal = (n - 1) * n + (n - 1)

    # Directions: up, down, left, right
    dx = [-1, 1, 0, 0]
    dy = [0, 0, -1, 1]

    while heap:
        # Pop minimum (simple extraction sort - find min)
        min_idx = 0
        for idx in range(1, len(heap)):
            if heap[idx][0] < heap[min_idx][0]:
                min_idx = idx
        f, current = heap[min_idx]
        heap[min_idx] = heap[-1]
        heap.pop()

        if in_closed[current]:
            continue

        in_closed[current] = True
        nodes_explored += 1

        if current == goal:
            break

        ci = current // n
        cj = current % n
        current_g = g_score[current]

        for d in range(4):
            ni = ci + dx[d]
            nj = cj + dy[d]
            if 0 <= ni < n and 0 <= nj < n:
                neighbor = ni * n + nj
                if not blocked[neighbor] and not in_closed[neighbor]:
                    tentative_g = current_g + 1
                    if tentative_g < g_score[neighbor]:
                        g_score[neighbor] = tentative_g
                        came_from[neighbor] = current
                        h = abs(ni - (n - 1)) + abs(nj - (n - 1))
                        heap.append((tentative_g + h, neighbor))

    if g_score[goal] == INF:
        return (-1, nodes_explored, -1)

    # Reconstruct path to find length and midpoint
    path_length = g_score[goal]
    # Walk backward to find midpoint
    path = [0] * (path_length + 1)
    pos = goal
    idx = path_length
    while pos != -1:
        path[idx] = pos
        pos = came_from[pos]
        idx -= 1

    mid_node = path[path_length // 2]
    path_midpoint_x = mid_node // n

    return (path_length, nodes_explored, path_midpoint_x)
