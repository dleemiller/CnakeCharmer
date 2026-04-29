"""Mean/Laplacian/convolution kernel operations for 2D/3D images."""

from __future__ import annotations


def convolve_mean2(image):
    h = len(image)
    w = len(image[0]) if h else 0
    out = [[0.0 for _ in range(w - 2)] for _ in range(h - 2)]
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            out[i - 1][j - 1] = 0.25 * (
                image[i - 1][j] + image[i + 1][j] + image[i][j - 1] + image[i][j + 1]
            )
    return out


def convolve_matrix2(image, kernel):
    h = len(image)
    w = len(image[0]) if h else 0
    ny = len(kernel)
    nx = len(kernel[0]) if ny else 0
    hy = ny // 2
    hx = nx // 2
    out = [[0.0 for _ in range(w - nx + 1)] for _ in range(h - ny + 1)]

    for y in range(hy, h - hy):
        for x in range(hx, w - hx):
            s = 0.0
            for ky in range(-hy, ny - hy):
                for kx in range(-hx, nx - hx):
                    s += kernel[ky + hy][kx + hx] * image[y + ky][x + kx]
            out[y - hy][x - hx] = abs(s)
    return out
