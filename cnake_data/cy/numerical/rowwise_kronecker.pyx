# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Row-wise Kronecker product of two matrices (Cython-optimized).

Keywords: kronecker, outer product, matrix, linear algebra, numerical, cython, benchmark
"""

from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(200000,))
def rowwise_kronecker(int n):
    """Compute row-wise Kronecker product for n rows with d=8, r=6."""
    cdef int d = 8
    cdef int r = 6
    cdef int i, j, k
    cdef double x_val, zv_val, val
    cdef double total_sum = 0.0
    cdef double max_value = -1e300

    for i in range(n):
        for j in range(d):
            x_val = ((i * 7 + j * 13 + 3) % 997) / 100.0 - 5.0
            for k in range(r):
                zv_val = ((i * 11 + k * 17 + 7) % 991) / 100.0 - 5.0
                val = x_val * zv_val
                total_sum += val
                if val > max_value:
                    max_value = val

    return (total_sum, max_value)
