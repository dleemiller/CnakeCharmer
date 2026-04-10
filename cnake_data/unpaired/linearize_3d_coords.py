def linearize_3d_coords(nx, ny, nz):
    """Compute the row-major linear index for every cell in a 3D grid.

    Given grid dimensions (nx, ny, nz), return a flat list of linear
    addresses for all (x, y, z) coordinates traversed in x-major,
    then y, then z order.  The linearization formula is:

        addr(x, y, z) = x + nx * (y + ny * z)

    This is useful for converting 3D voxel coordinates into flat array
    offsets for storage or lookup.

    Args:
        nx: size along x-axis (int).
        ny: size along y-axis (int).
        nz: size along z-axis (int).

    Returns:
        A list of int linear indices, one per grid cell, in
        (x, y, z) iteration order.
    """
    result = []
    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                result.append(x + nx * (y + ny * z))
    return result
