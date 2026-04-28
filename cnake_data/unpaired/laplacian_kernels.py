import numpy as np


def laplacian_cython(image):
    h, w = image.shape
    laplacian = np.empty((h - 2, w - 2), dtype=np.double)
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            laplacian[i - 1, j - 1] = (
                image[i - 1, j]
                + image[i + 1, j]
                + image[i, j - 1]
                + image[i, j + 1]
                - 4 * image[i, j]
            )
    return np.abs(laplacian) > 0.05


def laplacian_cython_bis(image):
    h, w = image.shape
    laplacian = np.empty((h - 2, w - 2), dtype=np.double)
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            l = (
                image[i - 1, j]
                + image[i + 1, j]
                + image[i, j - 1]
                + image[i, j + 1]
                - 4 * image[i, j]
            )
            laplacian[i - 1, j - 1] = 1 if l > 0.05 else 0
    return laplacian
