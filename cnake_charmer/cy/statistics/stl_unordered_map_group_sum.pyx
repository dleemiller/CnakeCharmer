# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Group-by-key aggregation using a C++ unordered_map.

Keywords: group by, aggregation, unordered_map, libcpp, hash map, benchmark
"""

from libcpp.unordered_map cimport unordered_map
from libcpp.utility cimport pair
from cython.operator cimport dereference, preincrement
from cnake_charmer.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(1000000,))
def stl_unordered_map_group_sum(int n):
    """Aggregate values into groups using an unordered_map and return statistics.

    Keys and values are generated deterministically:
        num_groups = n // 10
        key_i   = (i * 2654435761) % num_groups
        value_i = (i * 1103515245 + 12345) % 1000

    Args:
        n: Number of (key, value) pairs to process.

    Returns:
        Tuple of (xor_of_all_group_sums, max_group_sum).
    """
    cdef unordered_map[int, long long] groups
    groups.reserve(n // 10 + 10)

    cdef int i, key, value, num_groups
    cdef long long lv
    num_groups = n // 10

    for i in range(n):
        key = <int>((<long long>i * <long long>2654435761) % num_groups)
        lv = (<long long>i * <long long>1103515245 + <long long>12345) % <long long>1000
        value = <int>lv
        groups[key] += value

    cdef long long xor_sum = 0
    cdef long long max_sum = 0
    cdef long long s
    cdef unordered_map[int, long long].iterator it = groups.begin()
    cdef pair[int, long long] kv

    while it != groups.end():
        kv = dereference(it)
        s = kv.second
        xor_sum ^= s
        if s > max_sum:
            max_sum = s
        preincrement(it)

    return (<long long>xor_sum, <long long>max_sum)
