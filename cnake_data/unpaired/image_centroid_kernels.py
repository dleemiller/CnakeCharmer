import random


def exhaustive_centroid(img, r, g, b):
    """Compute centroid and count of pixels matching RGB color."""
    h = len(img)
    w = len(img[0]) if h else 0
    hmid = h // 2
    wmid = w // 2

    xc = 0.0
    yc = 0.0
    count = 0.0

    for x in range(w):
        for y in range(h):
            if img[y][x][0] == r and img[y][x][1] == g and img[y][x][2] == b:
                count += 1.0
                xc += (x - wmid - xc) / count
                yc += (y - hmid - yc) / count

    return [xc + wmid, yc + hmid], count


def diffusive_centroid(img, x0, y0, r, g, b, walk_steps=200):
    """Estimate centroid by random walks from interior seed point."""
    h = len(img)
    w = len(img[0]) if h else 0

    chaincount = 4
    jumping_stddev = 10

    xc = 0.0
    yc = 0.0

    for _ in range(chaincount):
        x = 0
        y = 0
        xc1 = float(x)
        yc1 = float(y)
        count = 1.0

        for _ in range(walk_steps):
            x1 = x + int(round(random.gauss(0, jumping_stddev)))
            y1 = y + int(round(random.gauss(0, jumping_stddev)))
            xx = x1 + x0
            yy = y1 + y0

            if xx < 0 or yy < 0 or xx >= w or yy >= h:
                continue

            if img[yy][xx][0] == r and img[yy][xx][1] == g and img[yy][xx][2] == b:
                x, y = x1, y1
                count += 1.0
                xc1 += (float(x) - xc1) / count
                yc1 += (float(y) - yc1) / count

        xc += xc1 / chaincount
        yc += yc1 / chaincount

    return [xc + x0, yc + y0]
