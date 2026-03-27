# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
# distutils: language = c++
"""Sliding window k-th largest using a C++ sorted vector with binary search.

Keywords: sliding window, k-th largest, sorted vector, libcpp, algorithm, benchmark
"""

from libcpp.vector cimport vector
from libcpp.algorithm cimport lower_bound
from cnake_charmer.benchmarks import cython_benchmark

DEF MOD = 1000000007
DEF W = 1000
DEF K = 100


@cython_benchmark(syntax="cy", args=(100000,))
def stl_set_kth_largest(int n):
    """Sliding window k-th largest using a C++ sorted vector.

    Values are generated as: val_i = (i * 2654435761) % 1_000_000_000
    Window size w=1000, k=100 (k-th largest = element at index w-k from
    the front of the sorted ascending window).

    For each position i >= w-1, insert val_i, remove the oldest value, and
    record the k-th largest element.

    Args:
        n: Total number of values to process.

    Returns:
        Tuple of (sum_of_kth_values % (10**9+7), final_kth_value).
    """
    cdef vector[long long] vals
    vals.resize(n)
    cdef int i
    for i in range(n):
        vals[i] = (<long long>i * <long long>2654435761) % <long long>1000000000

    cdef vector[long long] window
    window.reserve(W + 1)

    cdef long long sum_kth = 0
    cdef long long final_kth = 0
    cdef long long v, old_v
    cdef int pos, wsize

    for i in range(n):
        v = vals[i]
        # Insert v into sorted window using lower_bound
        pos = lower_bound(window.begin(), window.end(), v) - window.begin()
        window.insert(window.begin() + pos, v)

        if i >= W:
            # Remove oldest element vals[i - W]
            old_v = vals[i - W]
            pos = lower_bound(window.begin(), window.end(), old_v) - window.begin()
            window.erase(window.begin() + pos)

        if i >= W - 1:
            wsize = window.size()
            final_kth = window[wsize - K]
            sum_kth = (sum_kth + final_kth) % MOD

    return (<long long>sum_kth, <long long>final_kth)
