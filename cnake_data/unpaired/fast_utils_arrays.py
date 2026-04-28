import numpy as np


def make_nd_array(arrays=None):
    """Create a 2D array with stacked 1D arrays as columns."""
    if arrays is None:
        arrays = []
    na = len(arrays)
    if na == 0:
        return None

    a_size = len(arrays[0])
    for arr in arrays:
        if len(arr) != a_size:
            return None

    a = np.zeros((a_size, na), dtype=float)
    for i, arr in enumerate(arrays):
        for j in range(a_size):
            a[j, i] = arr[j]
    return a


def arange_long(start, stop=-1):
    """Range helper matching the cython utility semantics."""
    if stop == -1:
        return list(range(start))
    return list(range(start, stop))
