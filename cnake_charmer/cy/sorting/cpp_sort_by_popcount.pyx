# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
# distutils: language = c++
"""Sort integers by population count using std::sort with a custom C++ comparator.

Keywords: sorting, popcount, bit count, std::sort, comparator, cppclass, cython, benchmark
"""

from libcpp.vector cimport vector
from cnake_charmer.benchmarks import cython_benchmark

cdef extern from *:
    """
    #include <algorithm>
    struct PopcountCmp {
        bool operator()(int a, int b) const {
            int pa = __builtin_popcount((unsigned int)a);
            int pb = __builtin_popcount((unsigned int)b);
            return pa < pb || (pa == pb && a < b);
        }
    };
    """
    cdef cppclass PopcountCmp:
        pass

cdef extern from "<algorithm>" namespace "std" nogil:
    void sort[Iter, Comp](Iter first, Iter last, Comp comp)

DEF MOD = 1000000007
DEF LIMIT = 1000000000


@cython_benchmark(syntax="cy", args=(300000,))
def cpp_sort_by_popcount(int n):
    """Generate n values, sort by (popcount, value), return position-weighted hash.

    Values: val[i] = (i * 2654435761) % (10**9)
    Sorted with C++ custom comparator using __builtin_popcount.
    Hash: sum(arr[i] * (i+1)) % (10**9+7)

    Args:
        n: Number of values to generate and sort.

    Returns:
        Position-weighted hash of sorted array.
    """
    cdef vector[int] arr
    cdef int i
    cdef long long li

    arr.resize(n)
    for i in range(n):
        li = (<long long>i * <long long>2654435761) % LIMIT
        arr[i] = <int>li

    cdef PopcountCmp cmp
    sort(arr.begin(), arr.end(), cmp)

    cdef long long result = 0
    for i in range(n):
        result = (result + <long long>arr[i] * (i + 1)) % MOD

    return result
