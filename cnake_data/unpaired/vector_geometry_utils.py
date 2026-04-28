import math


def isleft(pt0, pt1, pt2):
    return (pt1[0] - pt0[0]) * (pt2[1] - pt0[1]) - (pt1[1] - pt0[1]) * (pt2[0] - pt0[0]) > 0.0


def polarangle(pt0, pt1):
    return math.atan2(pt1[1] - pt0[1], pt1[0] - pt0[0])


def pt_nearest_planar(x, y, endpt0_0, endpt0_1, endpt1_0, endpt1_1):
    """Nearest point on segment to point (x, y), with distance."""
    pt0 = (endpt0_0, endpt0_1)
    pt1 = (endpt1_0, endpt1_1)

    ux = x - pt0[0]
    uy = y - pt0[1]
    vx = pt1[0] - pt0[0]
    vy = pt1[1] - pt0[1]

    vv = vx * vx + vy * vy
    if vv == 0.0:
        d = math.hypot(x - pt0[0], y - pt0[1])
        return pt0, d

    t = (ux * vx + uy * vy) / vv
    t = min(max(t, 0.0), 1.0)
    proj = (pt0[0] + t * vx, pt0[1] + t * vy)
    d = math.hypot(x - proj[0], y - proj[1])
    return proj, d


def bbox_intersection_area(bb0, bb1):
    dx = max(min(bb0[2], bb1[2]) - max(bb0[0], bb1[0]), 0.0)
    dy = max(min(bb0[3], bb1[3]) - max(bb0[1], bb1[1]), 0.0)
    return dx * dy


def iswithin(bbox, pt):
    return bbox[0] <= pt[0] < bbox[2] and bbox[1] <= pt[1] < bbox[3]


def hashpt(xmin, ymin, xmax, ymax, x, y):
    """Yield successive quadtree quadrants (0..3) for point (x, y)."""
    while True:
        xm = 0.5 * (xmin + xmax)
        ym = 0.5 * (ymin + ymax)

        if x < xm and y < ym:
            geohash = 0
            xmax, ymax = xm, ym
        elif x >= xm and y < ym:
            geohash = 1
            xmin, ymax = xm, ym
        elif x < xm and y >= ym:
            geohash = 2
            xmax, ymin = xm, ym
        else:
            geohash = 3
            xmin, ymin = xm, ym

        yield geohash
