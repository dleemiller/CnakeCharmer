# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Parse comma-separated mixed tokens and summarize numeric invoices (Cython).

Adapted from The Stack v2 Cython candidate:
- blob_id: d24c7ac24a0676b6e56e0afebdde3fd57041433e
- filename: proWrd.pyx
"""

from cnake_data.benchmarks import cython_benchmark


cdef void _invoice_kernel(
    int n_fields,
    int* count_out,
    unsigned int* total_out,
    int* first_out,
    int* last_out,
) noexcept nogil:
    cdef unsigned int state = 42424242
    cdef unsigned int mask = 0xFFFFFFFF
    cdef int i, nd
    cdef int count = 0
    cdef unsigned int total = 0
    cdef int first = -1
    cdef int last = -1
    cdef int val

    for i in range(n_fields):
        state = (1664525 * state + 1013904223) & mask
        if (state & 3) < 2:
            val = <int>((state >> 8) % 100000)
            if count == 0:
                first = val
            last = val
            count += 1
            total += <unsigned int>val
        else:
            nd = <int>((state >> 11) % 1000)
            nd += 1

    count_out[0] = count
    total_out[0] = total & mask
    first_out[0] = first
    last_out[0] = last


@cython_benchmark(syntax="cy", args=(50000,))
def stack_invoice_digit_sum(int n_fields):
    cdef int count = 0
    cdef unsigned int total = 0
    cdef int first = -1
    cdef int last = -1

    with nogil:
        _invoice_kernel(n_fields, &count, &total, &first, &last)

    return (count, total, first, last)
