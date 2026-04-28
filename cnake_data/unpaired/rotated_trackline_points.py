import math


def rotated_trackline_points(easting, northing, y_values, depth, theta):
    """Generate rotated/translated points around a center line.

    Args:
        easting: Center x-coordinate.
        northing: Center y-coordinate.
        y_values: Iterable of slant ranges (must satisfy y^2 >= depth^2).
        depth: Depth offset.
        theta: Rotation angle in radians.

    Returns:
        Tuple (xx, yy) lists for transformed coordinates.
    """
    sin_t = math.sin(theta)
    cos_t = math.cos(theta)
    dist_y = depth * cos_t
    dist_x = depth * sin_t
    d_sq = depth * depth

    rangedist = []
    for y in y_values:
        v = y * y - d_sq
        if v < 0.0:
            v = 0.0
        rangedist.append(math.sqrt(v))

    x = [easting] * len(rangedist) + [easting] * len(rangedist)
    y = [northing + r for r in rangedist] + [northing - r for r in rangedist]

    xx = []
    yy = []
    for xi, yi in zip(x, y, strict=False):
        xr = easting - ((xi - easting) * cos_t) - ((yi - northing) * sin_t)
        yr = northing - ((xi - easting) * sin_t) + ((yi - northing) * cos_t)
        xx.append(xr + dist_x)
        yy.append(yr + dist_y)

    return xx, yy
