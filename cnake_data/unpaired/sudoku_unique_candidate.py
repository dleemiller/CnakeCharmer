def candidate_unique(grid, area_idxs, candidate):
    """Set cell if candidate appears exactly once in indexed area."""
    candidate_count = 0
    last_r = last_c = -1

    for r, c in area_idxs:
        if grid[r][c][candidate]:
            candidate_count += 1
            last_r, last_c = r, c

    if candidate_count == 1:
        # set_cell behavior: clear others, set chosen candidate
        for k in range(1, 10):
            grid[last_r][last_c][k] = 1 if k == candidate else 0
        return 1
    return 0


def unique(grid, row_fn, col_fn, block_fn):
    """Apply unique-candidate rule across rows, cols, and blocks."""
    for candidate in range(1, 10):
        for r in range(9):
            if candidate_unique(grid, row_fn(r), candidate):
                return 1
        for c in range(9):
            if candidate_unique(grid, col_fn(c), candidate):
                return 1
        for b in range(9):
            if candidate_unique(grid, block_fn(b), candidate):
                return 1
    return 0
