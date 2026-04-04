"""Dateline-aware bounding box of a geodesic path.

Keywords: geometry, bounding box, dateline, longitude, latitude, geodesic
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(20000,))
def dateline_bbox(n: int) -> tuple[int, int, int, int]:
    """Compute the dateline-aware bounding box of n coordinate pairs.

    Generates n (lon, lat) pairs deterministically:
      lon[i] = ((i * 137.5) % 360) - 180
      lat[i] = ((i * 73.1) % 180) - 90

    Detects dateline crossings and adjusts xmin/xmax accordingly.

    Args:
        n: Number of coordinate pairs.

    Returns:
        (xmin, ymin, xmax, ymax) each scaled by 1000 and truncated to int.
    """
    lons = [((i * 137.5) % 360) - 180 for i in range(n)]
    lats = [((i * 73.1) % 180) - 90 for i in range(n)]

    xmin = xmax = lons[0]
    ymin = ymax = lats[0]
    rot = 0.0

    for i in range(n - 1):
        x0, x1 = lons[i], lons[i + 1]
        ymin = min(ymin, lats[i + 1])
        ymax = max(ymax, lats[i + 1])

        s0 = 1 if x0 >= 0 else -1
        s1 = 1 if x1 >= 0 else -1
        if s0 != s1 and abs(x0 - x1) > 180.0:
            xdateline = 1 if x1 - x0 > 180 else -1
            rot -= xdateline * 360.0
            adj = x1 + rot
            xmin = min(xmin, adj)
            xmax = max(xmax, adj)
        else:
            if x0 > x1:
                xmin = min(xmin, x1)
            else:
                xmax = max(xmax, x1)

    xmin = (xmin + 180) % 360 - 180
    xmax = (xmax + 180) % 360 - 180
    return (int(xmin * 1000), int(ymin * 1000), int(xmax * 1000), int(ymax * 1000))
