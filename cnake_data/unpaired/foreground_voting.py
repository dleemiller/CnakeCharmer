def foreground_voting(n):
    """Classify groups as foreground/background using a voting scheme on a 2D grid.

    Creates an n x n grid where each cell has a foreground flag and a group ID.
    Groups are determined by spatial tiling. A group is classified as foreground
    if more than 60% of its cells are flagged and its size exceeds a threshold.
    Returns statistics about the classification.

    Args:
        n: Grid dimension (n x n).

    Returns:
        (num_foreground_groups, num_masked_cells, total_votes) as a tuple.
    """
    tile_size = max(1, n // 4)
    num_groups = ((n + tile_size - 1) // tile_size) ** 2

    # Build foreground flags and group assignments
    foreground = [[0] * n for _ in range(n)]
    groups = [[0] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            gi = i // tile_size
            gj = j // tile_size
            groups[i][j] = gi * ((n + tile_size - 1) // tile_size) + gj
            # Deterministic foreground pattern
            if (i * 7 + j * 13 + 3) % 5 < 3:
                foreground[i][j] = 1

    # Count votes per group
    group_size = [0] * num_groups
    votes = [0] * num_groups

    for i in range(n):
        for j in range(n):
            g = groups[i][j]
            group_size[g] += 1
            if foreground[i][j] == 1:
                votes[g] += 1

    # Classify groups
    fg_groups = [0] * num_groups
    for g in range(num_groups):
        if group_size[g] > 2 and votes[g] / group_size[g] > 0.6:
            fg_groups[g] = 1

    # Build mask
    num_foreground = sum(fg_groups)
    num_masked = 0
    for i in range(n):
        for j in range(n):
            if fg_groups[groups[i][j]] == 1:
                num_masked += 1

    total_votes = sum(votes)

    return (num_foreground, num_masked, total_votes)
