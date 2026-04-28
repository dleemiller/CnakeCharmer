import math


def pad_grayscale(img, padding):
    """Zero-pad a 2D grayscale image."""
    h = len(img)
    w = len(img[0]) if h else 0
    out_h = h + 2 * padding
    out_w = w + 2 * padding
    out = [[0 for _ in range(out_w)] for _ in range(out_h)]

    for i in range(h):
        for j in range(w):
            out[i + padding][j + padding] = int(img[i][j])
    return out


def pad_colored(img, padding):
    """Zero-pad a 3D RGB image."""
    h = len(img)
    w = len(img[0]) if h else 0
    c = len(img[0][0]) if h and w else 0
    out_h = h + 2 * padding
    out_w = w + 2 * padding
    out = [[[0 for _ in range(c)] for _ in range(out_w)] for _ in range(out_h)]

    for i in range(h):
        for j in range(w):
            for k in range(c):
                out[i + padding][j + padding][k] = int(img[i][j][k])
    return out


def multiply_and_sum2d(image_extract, coeff):
    """Compute scaled sum of elementwise product for 2D arrays."""
    total = 0
    for i in range(len(image_extract)):
        for j in range(len(image_extract[0])):
            total += int(image_extract[i][j]) * int(coeff[i][j])
    return total / 100000000.0


def multiply_and_sum3d(image_extract, coeff):
    """Compute scaled per-channel sum of elementwise products."""
    c = len(image_extract[0][0]) if image_extract else 0
    totals = [0 for _ in range(c)]

    for i in range(len(image_extract)):
        for j in range(len(image_extract[0])):
            w = int(coeff[i][j])
            for k in range(c):
                totals[k] += int(image_extract[i][j][k]) * w

    return [x / 100000000.0 for x in totals]


def gauss(sigma, x, y, mean=0):
    """Evaluate isotropic Gaussian over radial distance from (x, y)."""
    d = math.sqrt(x * x + y * y)
    return math.exp(-((d - mean) ** 2) / (2 * sigma * sigma)) / math.sqrt(
        2 * math.pi * sigma * sigma
    )


def gausskernel(width, loc, sigma):
    """Create a width x width Gaussian kernel shifted by fractional loc."""
    k = [[0.0 for _ in range(width)] for _ in range(width)]
    shift = (float(width) - 1.0) / 2.0

    dx = loc[0] - int(loc[0])
    dy = loc[1] - int(loc[1])

    for x in range(width):
        for y in range(width):
            k[y][x] = gauss(sigma, (x - shift) - dx, (y - shift) - dy, 0)

    return k
