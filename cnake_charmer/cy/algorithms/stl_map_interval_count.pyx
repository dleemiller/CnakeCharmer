# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Online frequency counting with sorted min/max via C++ map.

Uses std::map for O(log n) insert/erase and O(1) begin/rbegin for
min/max access. Python's bisect.insort is O(n) per insert due to
list shifting; list.pop(idx) is also O(n).

Keywords: algorithms, sorted map, online, frequency, libcpp, map, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from libcpp.map cimport map as cpp_map
from cython.operator cimport dereference, predecrement
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(200000,))
def stl_map_interval_count(int n):
    """Insert/remove keys in a sliding window, tracking min/max spread.

    std::map gives O(log n) insert/erase and O(1) min/max via
    begin()/rbegin(). Python needs O(n) bisect.insort + O(n) list.pop.

    Args:
        n: Number of operations.

    Returns:
        Tuple of (final_distinct_count, spread_sum).
    """
    cdef cpp_map[int, int] freq
    cdef int i, key, old_key
    cdef long long spread_sum = 0
    cdef int key_range = n * 4
    cdef long long li
    cdef int *window = NULL
    cdef int min_key, max_key
    cdef cpp_map[int, int].iterator it

    window = <int *>malloc(n * sizeof(int))
    if not window:
        raise MemoryError()

    for i in range(n):
        li = <long long>i * <long long>2654435761
        key = <int>(li % key_range)
        window[i] = key
        freq[key] += 1

        # Sliding window: remove oldest every 8 steps
        if i >= 8 and i & 7 == 0:
            old_key = window[i - 8]
            freq[old_key] -= 1
            if freq[old_key] == 0:
                freq.erase(freq.find(old_key))

        if not freq.empty():
            min_key = dereference(freq.begin()).first
            it = freq.end()
            predecrement(it)
            max_key = dereference(it).first
            spread_sum += max_key - min_key

    free(window)
    return (<int>freq.size(), spread_sum)
