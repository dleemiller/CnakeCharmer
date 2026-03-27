# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""
Naive suffix array construction (Cython-optimized).

Keywords: string processing, suffix array, sorting, naive, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from libc.string cimport memcmp
from cnake_charmer.benchmarks import cython_benchmark


cdef inline int _compare_suffixes(const char *s, int n, int a, int b) noexcept nogil:
    """Compare suffix starting at a with suffix starting at b."""
    cdef int la = n - a
    cdef int lb = n - b
    cdef int min_len = la if la < lb else lb
    cdef int cmp_result = memcmp(&s[a], &s[b], min_len)
    if cmp_result != 0:
        return cmp_result
    return la - lb


cdef void _insertion_sort(int *sa, const char *s, int n, int lo, int hi) noexcept nogil:
    """Insertion sort for small ranges."""
    cdef int i, j, key_val
    for i in range(lo + 1, hi):
        key_val = sa[i]
        j = i - 1
        while j >= lo and _compare_suffixes(s, n, sa[j], key_val) > 0:
            sa[j + 1] = sa[j]
            j -= 1
        sa[j + 1] = key_val


cdef void _quicksort(int *sa, const char *s, int n, int lo, int hi) noexcept nogil:
    """Quicksort with true median-of-three pivot selection."""
    cdef int i, j, mid, tmp, pivot_val
    while hi - lo > 16:
        mid = (lo + hi) >> 1

        # True median-of-three: sort lo, mid, hi-1 so median ends up at mid
        if _compare_suffixes(s, n, sa[lo], sa[mid]) > 0:
            tmp = sa[lo]; sa[lo] = sa[mid]; sa[mid] = tmp
        if _compare_suffixes(s, n, sa[lo], sa[hi - 1]) > 0:
            tmp = sa[lo]; sa[lo] = sa[hi - 1]; sa[hi - 1] = tmp
        if _compare_suffixes(s, n, sa[mid], sa[hi - 1]) > 0:
            tmp = sa[mid]; sa[mid] = sa[hi - 1]; sa[hi - 1] = tmp
        # Now sa[lo] <= sa[mid] <= sa[hi-1]; use mid as pivot
        pivot_val = sa[mid]

        i = lo
        j = hi - 1
        while True:
            while _compare_suffixes(s, n, sa[i], pivot_val) < 0:
                i += 1
            while _compare_suffixes(s, n, sa[j], pivot_val) > 0:
                j -= 1
            if i >= j:
                break
            tmp = sa[i]; sa[i] = sa[j]; sa[j] = tmp
            i += 1
            j -= 1

        # Tail-call optimization: recurse on smaller half, iterate on larger
        if j - lo < hi - i:
            _quicksort(sa, s, n, lo, j + 1)
            lo = i
        else:
            _quicksort(sa, s, n, i, hi)
            hi = j + 1
    _insertion_sort(sa, s, n, lo, hi)


@cython_benchmark(syntax="cy", args=(10000,))
def suffix_array_naive(int n):
    """Build a suffix array and return sum of first 100 positions.

    Args:
        n: Length of the string.

    Returns:
        Sum of first min(100, n) suffix array positions.
    """
    cdef int i, limit
    cdef long long total = 0

    # Build deterministic string as bytes
    cdef char *s = <char *>malloc(n * sizeof(char))
    if s == NULL:
        raise MemoryError("Failed to allocate string buffer")

    cdef int *sa = <int *>malloc(n * sizeof(int))
    if sa == NULL:
        free(s)
        raise MemoryError("Failed to allocate suffix array")

    # Initialize both buffers in a single pass
    for i in range(n):
        s[i] = 65 + (i * 7 + 3) % 26
        sa[i] = i

    # Sort using quicksort with suffix comparison (GIL released)
    with nogil:
        _quicksort(sa, s, n, 0, n)

    # Sum first 100 positions
    limit = 100 if n > 100 else n
    for i in range(limit):
        total += sa[i]

    free(sa)
    free(s)
    return total
