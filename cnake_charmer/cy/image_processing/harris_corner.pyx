# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Harris corner detector response on a deterministic image (Cython-optimized).

Keywords: image processing, harris, corner detection, gradient, response, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(300,))
def harris_corner(int n):
    """Compute Harris corner response on n x n deterministic image with C arrays."""
    cdef double *image = <double *>malloc(n * n * sizeof(double))
    cdef double *Ix = <double *>malloc(n * n * sizeof(double))
    cdef double *Iy = <double *>malloc(n * n * sizeof(double))

    if not image or not Ix or not Iy:
        free(image); free(Ix); free(Iy)
        raise MemoryError()

    cdef int i, j, wi, wj
    cdef double gx, gy, ix_val, iy_val
    cdef double sxx, syy, sxy, det, trace, R
    cdef double k = 0.04
    cdef double threshold = 1e6
    cdef int num_corners = 0
    cdef double max_response = -1e30
    cdef double response_sum = 0.0
    cdef int val
    cdef double noise

    # Generate image
    for i in range(n):
        for j in range(n):
            val = ((i // 16) + (j // 16)) % 2
            noise = ((i * 7 + j * 13 + 3) % 17) / 17.0
            image[i * n + j] = val * 200.0 + noise * 50.0

    # Zero out gradient arrays
    for i in range(n * n):
        Ix[i] = 0.0
        Iy[i] = 0.0

    # Compute Sobel gradients
    for i in range(1, n - 1):
        for j in range(1, n - 1):
            gx = (-1.0 * image[(i - 1) * n + (j - 1)]
                  + 1.0 * image[(i - 1) * n + (j + 1)]
                  - 2.0 * image[i * n + (j - 1)]
                  + 2.0 * image[i * n + (j + 1)]
                  - 1.0 * image[(i + 1) * n + (j - 1)]
                  + 1.0 * image[(i + 1) * n + (j + 1)])
            gy = (-1.0 * image[(i - 1) * n + (j - 1)]
                  - 2.0 * image[(i - 1) * n + j]
                  - 1.0 * image[(i - 1) * n + (j + 1)]
                  + 1.0 * image[(i + 1) * n + (j - 1)]
                  + 2.0 * image[(i + 1) * n + j]
                  + 1.0 * image[(i + 1) * n + (j + 1)])
            Ix[i * n + j] = gx
            Iy[i * n + j] = gy

    # Harris response
    for i in range(2, n - 2):
        for j in range(2, n - 2):
            sxx = 0.0
            syy = 0.0
            sxy = 0.0
            for wi in range(-1, 2):
                for wj in range(-1, 2):
                    ix_val = Ix[(i + wi) * n + (j + wj)]
                    iy_val = Iy[(i + wi) * n + (j + wj)]
                    sxx += ix_val * ix_val
                    syy += iy_val * iy_val
                    sxy += ix_val * iy_val

            det = sxx * syy - sxy * sxy
            trace = sxx + syy
            R = det - k * trace * trace

            response_sum += R
            if R > max_response:
                max_response = R
            if R > threshold:
                num_corners += 1

    free(image)
    free(Ix)
    free(Iy)

    return (num_corners, max_response, response_sum)
