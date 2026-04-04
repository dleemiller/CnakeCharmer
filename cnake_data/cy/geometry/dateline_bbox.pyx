# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Dateline-aware bounding box of a geodesic path (Cython-optimized)."""

from libc.stdlib cimport malloc, free
from libc.math cimport fmod
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(20000,))
def dateline_bbox(int n):
    """Compute dateline-aware bounding box using C arrays.

    Args:
        n: Number of coordinate pairs.

    Returns:
        (xmin, ymin, xmax, ymax) each scaled by 1000 and truncated to int.
    """
    cdef int i
    cdef double x0, x1, y1, adj, rot
    cdef double xmin, xmax, ymin, ymax
    cdef int s0, s1, xdateline
    cdef double xmin_out, xmax_out

    cdef double *lons = <double *>malloc(n * sizeof(double))
    cdef double *lats = <double *>malloc(n * sizeof(double))

    if lons == NULL or lats == NULL:
        if lons != NULL: free(lons)
        if lats != NULL: free(lats)
        raise MemoryError("Failed to allocate coordinate arrays")

    with nogil:
        for i in range(n):
            lons[i] = fmod(i * 137.5, 360.0) - 180.0
            lats[i] = fmod(i * 73.1, 180.0) - 90.0

    xmin = lons[0]
    xmax = lons[0]
    ymin = lats[0]
    ymax = lats[0]
    rot = 0.0

    with nogil:
        for i in range(n - 1):
            x0 = lons[i]
            x1 = lons[i + 1]
            y1 = lats[i + 1]

            if y1 < ymin:
                ymin = y1
            if y1 > ymax:
                ymax = y1

            s0 = 1 if x0 >= 0.0 else -1
            s1 = 1 if x1 >= 0.0 else -1

            if s0 != s1 and (x0 - x1 > 180.0 or x1 - x0 > 180.0):
                if x1 - x0 > 180.0:
                    xdateline = 1
                else:
                    xdateline = -1
                rot -= xdateline * 360.0
                adj = x1 + rot
                if adj < xmin:
                    xmin = adj
                if adj > xmax:
                    xmax = adj
            else:
                if x0 > x1:
                    if x1 < xmin:
                        xmin = x1
                else:
                    if x1 > xmax:
                        xmax = x1

    xmin_out = fmod(xmin + 180.0, 360.0) - 180.0
    xmax_out = fmod(xmax + 180.0, 360.0) - 180.0

    free(lons)
    free(lats)

    return (int(xmin_out * 1000), int(ymin * 1000), int(xmax_out * 1000), int(ymax * 1000))
