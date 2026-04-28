def grid3d_to_nested_list(nx, ny, nz, values):
    """Build a dense 3D grid from flat values.

    Args:
        nx, ny, nz: Grid dimensions.
        values: Flat iterable with nx*ny*nz entries.

    Returns:
        3D list grid[x][y][z].
    """
    total = nx * ny * nz
    if len(values) != total:
        raise ValueError(f"Expected {total} values, got {len(values)}")

    out = [[[0.0 for _ in range(nz)] for _ in range(ny)] for _ in range(nx)]
    idx = 0
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                out[i][j][k] = values[idx]
                idx += 1
    return out
