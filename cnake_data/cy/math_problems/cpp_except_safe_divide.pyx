# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
# distutils: language = c++
"""Safe integer division using inline C++ function with except +ValueError.

Keywords: math, division, exception, ValueError, extern, c++, cython, benchmark
"""

from cnake_data.benchmarks import cython_benchmark

cdef extern from *:
    """
    #include <stdexcept>
    inline int safe_divide(int a, int b) {
        if (b == 0) {
            throw std::invalid_argument("division by zero");
        }
        return a / b;
    }
    """
    int safe_divide(int a, int b) except +ValueError


@cython_benchmark(syntax="cy", args=(500000,))
def cpp_except_safe_divide(int n):
    """Perform n integer divisions, catching ValueError when b==0.

    a[i] = (i * 2654435761) % 1000 - 500
    b[i] = (i * 1103515245) % 21 - 10  (range [-10, 10], includes 0)
    C++ safe_divide throws std::invalid_argument on b==0, mapped to ValueError.

    Args:
        n: Number of division operations.

    Returns:
        Tuple of (sum_of_valid_results, error_count).
    """
    cdef int i, a, b
    cdef long long li
    cdef long long sum_valid = 0
    cdef int error_count = 0

    for i in range(n):
        li = (<long long>i * <long long>2654435761) & <long long>0xFFFFFFFF
        a = <int>(li % 1000) - 500
        li = (<long long>i * <long long>1103515245) & <long long>0xFFFFFFFF
        b = <int>(li % 21) - 10
        try:
            sum_valid += safe_divide(a, b)
        except ValueError:
            error_count += 1

    return (sum_valid, error_count)
