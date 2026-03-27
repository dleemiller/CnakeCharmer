# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
# distutils: language = c++
"""Safe vector access with bounds checking via C++ at() and except +.

Keywords: algorithms, bounds check, exception handling, at(), vector, cppclass, cython, benchmark
"""

from cnake_charmer.benchmarks import cython_benchmark

cdef extern from "<vector>" namespace "std" nogil:
    cdef cppclass checked_vector "std::vector<int>":
        checked_vector()
        void push_back(int)
        int& at(size_t) except +
        size_t size()

DEF MOD = 1000000007


@cython_benchmark(syntax="cy", args=(200000,))
def cpp_except_bounds_check(int n):
    """Populate a C++ vector of size n and perform m=2*n lookups, some out of bounds.

    Values: val[i] = (i * 2654435761) % n
    Lookup indices: idx = (i * 1103515245 + 12345) % (n + n // 100)
    Uses vector::at() which throws std::out_of_range mapped to IndexError.

    Args:
        n: Size of the vector and half the number of lookups.

    Returns:
        Tuple of (sum_of_valid_values % (10**9+7), out_of_bounds_count).
    """
    cdef checked_vector data
    cdef int i
    cdef long long li
    cdef int m = 2 * n
    cdef int upper = n + n // 100
    cdef long long sum_valid = 0
    cdef int oob_count = 0
    cdef size_t idx

    for i in range(n):
        li = (<long long>i * <long long>2654435761) % n
        data.push_back(<int>li)

    for i in range(m):
        idx = (<long long>i * <long long>1103515245 + <long long>12345) % upper
        try:
            sum_valid = (sum_valid + data.at(idx)) % MOD
        except IndexError:
            oob_count += 1

    return (<long long>sum_valid, oob_count)
