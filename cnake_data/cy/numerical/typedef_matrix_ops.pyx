# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""4x4 matrix multiplication using ctypedef arrays (Cython-optimized).

Keywords: numerical, matrix, multiplication, linear algebra, ctypedef, cython, benchmark
"""

from cnake_data.benchmarks import cython_benchmark

ctypedef double[4] vec4_t
ctypedef double[16] mat4_t


cdef void _mat4_mul(double *a, double *b, double *out) noexcept:
    """Multiply two 4x4 matrices: out = a * b."""
    cdef int i, j, k
    cdef double s
    for i in range(4):
        for j in range(4):
            s = 0.0
            for k in range(4):
                s += a[i * 4 + k] * b[k * 4 + j]
            out[i * 4 + j] = s


@cython_benchmark(syntax="cy", args=(10000,))
def typedef_matrix_ops(int n):
    """Repeatedly multiply 4x4 matrices using ctypedef arrays."""
    cdef mat4_t a, b, result, temp
    cdef int i, j, it

    for i in range(4):
        for j in range(4):
            a[i * 4 + j] = ((i * 4 + j) * 7 + 3) % 17 / 10.0
            b[i * 4 + j] = ((i * 4 + j) * 11 + 5) % 19 / 10.0

    # Identity
    for i in range(16):
        result[i] = 0.0
    for i in range(4):
        result[i * 4 + i] = 1.0

    for it in range(n):
        _mat4_mul(result, a, temp)
        for i in range(16):
            result[i] = temp[i] * 0.1 + b[i] * 0.01

    cdef double trace = 0.0
    for i in range(4):
        trace += result[i * 4 + i]

    return trace
