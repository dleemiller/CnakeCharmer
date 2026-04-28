def init_grid(width, height, value=0):
    """Create a width x height grid filled with value."""
    return [[value for _ in range(width)] for _ in range(height)]


def mirror_grid_horiz(grid):
    """Mirror each row of a grid in place (left-right flip)."""
    h = len(grid)
    w = len(grid[0]) if h else 0
    for y in range(h):
        for x in range(w // 2):
            grid[y][x], grid[y][w - 1 - x] = grid[y][w - 1 - x], grid[y][x]


def rotate_grid_clockwise(grid):
    """Rotate a square grid 90 degrees clockwise."""
    n = len(grid)
    out = init_grid(n, n, 0)
    for y in range(n):
        for x in range(n):
            out[y][x] = grid[n - 1 - x][y]
    return out


def grids_equal(a, b):
    """Check elementwise equality of two same-shaped grids."""
    if len(a) != len(b):
        return False
    if a and len(a[0]) != len(b[0]):
        return False
    for y in range(len(a)):
        for x in range(len(a[0])):
            if a[y][x] != b[y][x]:
                return False
    return True
