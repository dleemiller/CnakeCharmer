# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Patience sort on a deterministic integer array (Cython-optimized).

Keywords: algorithms, patience sort, sorting, piles, cython, benchmark
"""

from libc.stdlib cimport malloc, free, realloc
from cnake_charmer.benchmarks import cython_benchmark


# ---------------------------------------------------------------------------
# C-level min-heap helpers (operate on two parallel int arrays)
# ---------------------------------------------------------------------------

cdef void heap_sift_down(int *hv, int *hp, int size, int i) nogil:
    """Sift element at index i down to restore min-heap property on hv."""
    cdef int left, right, smallest, tmp_v, tmp_p
    while True:
        left    = 2 * i + 1
        right   = 2 * i + 2
        smallest = i
        if left  < size and hv[left]  < hv[smallest]:
            smallest = left
        if right < size and hv[right] < hv[smallest]:
            smallest = right
        if smallest == i:
            break
        tmp_v = hv[i];  hv[i] = hv[smallest];  hv[smallest] = tmp_v
        tmp_p = hp[i];  hp[i] = hp[smallest];  hp[smallest] = tmp_p
        i = smallest


cdef void heap_sift_up(int *hv, int *hp, int i) nogil:
    """Sift element at index i up to restore min-heap property on hv."""
    cdef int parent, tmp_v, tmp_p
    while i > 0:
        parent = (i - 1) // 2
        if hv[i] < hv[parent]:
            tmp_v = hv[i];  hv[i] = hv[parent];  hv[parent] = tmp_v
            tmp_p = hp[i];  hp[i] = hp[parent];  hp[parent] = tmp_p
            i = parent
        else:
            break


# ---------------------------------------------------------------------------
# Main sort
# ---------------------------------------------------------------------------

@cython_benchmark(syntax="cy", args=(100000,))
def patience_sort(int n):
    """Sort array using patience sort with C arrays for piles and heap merge."""

    # --- input / output buffers ---
    cdef int *arr    = <int *>malloc(n * sizeof(int))
    cdef int *output = <int *>malloc(n * sizeof(int))
    if not arr or not output:
        if arr:    free(arr)
        if output: free(output)
        raise MemoryError()

    cdef int i

    # Generate input (no GIL needed, pure C)
    with nogil:
        for i in range(n):
            arr[i] = (i * 31 + 17) % n

    # --- pile metadata ---
    # piles[p]       : pointer to pile p's element array (grows dynamically)
    # pile_sizes[p]  : current number of elements in pile p
    # pile_caps[p]   : allocated capacity of pile p
    # pile_tops[p]   : top element of pile p (for binary search)
    #
    # We use int** here which requires GIL-compatible malloc; the realloc
    # calls also need the GIL, so the build phase runs with the GIL but
    # all arithmetic/comparisons are typed C.
    cdef int   num_piles  = 0
    cdef int **piles      = <int **>malloc(n * sizeof(int *))
    cdef int  *pile_sizes = <int  *>malloc(n * sizeof(int))
    cdef int  *pile_caps  = <int  *>malloc(n * sizeof(int))
    cdef int  *pile_tops  = <int  *>malloc(n * sizeof(int))

    if not piles or not pile_sizes or not pile_caps or not pile_tops:
        free(arr); free(output)
        if piles:      free(piles)
        if pile_sizes: free(pile_sizes)
        if pile_caps:  free(pile_caps)
        if pile_tops:  free(pile_tops)
        raise MemoryError()

    # --- heap storage ---
    cdef int *heap_v    = <int *>malloc(n * sizeof(int))
    cdef int *heap_p    = <int *>malloc(n * sizeof(int))
    cdef int  heap_size = 0

    if not heap_v or not heap_p:
        free(arr); free(output)
        free(piles); free(pile_sizes); free(pile_caps); free(pile_tops)
        if heap_v: free(heap_v)
        if heap_p: free(heap_p)
        raise MemoryError()

    cdef int     val, lo_idx, hi_idx, mid_idx
    cdef int     new_cap, pile_idx, top_val, out_idx
    cdef int    *new_ptr

    # --- build piles ---
    # realloc requires the GIL; all loop variables are C-typed so
    # the body is still compiled to near-zero Python overhead.
    for i in range(n):
        val = arr[i]

        # binary search: leftmost pile whose top >= val
        lo_idx = 0
        hi_idx = num_piles
        while lo_idx < hi_idx:
            mid_idx = (lo_idx + hi_idx) // 2
            if pile_tops[mid_idx] >= val:
                hi_idx = mid_idx
            else:
                lo_idx = mid_idx + 1

        if lo_idx == num_piles:
            # start a new pile with initial capacity 8
            new_ptr = <int *>malloc(8 * sizeof(int))
            if not new_ptr:
                # cleanup and raise
                for i in range(num_piles):
                    free(piles[i])
                free(piles); free(pile_sizes); free(pile_caps); free(pile_tops)
                free(heap_v); free(heap_p); free(arr); free(output)
                raise MemoryError()
            piles[num_piles]      = new_ptr
            pile_sizes[num_piles] = 0
            pile_caps[num_piles]  = 8
            num_piles += 1

        # grow pile if needed
        if pile_sizes[lo_idx] == pile_caps[lo_idx]:
            new_cap = pile_caps[lo_idx] * 2
            new_ptr = <int *>realloc(piles[lo_idx], new_cap * sizeof(int))
            if not new_ptr:
                for i in range(num_piles):
                    free(piles[i])
                free(piles); free(pile_sizes); free(pile_caps); free(pile_tops)
                free(heap_v); free(heap_p); free(arr); free(output)
                raise MemoryError()
            piles[lo_idx]     = new_ptr
            pile_caps[lo_idx] = new_cap

        piles[lo_idx][pile_sizes[lo_idx]] = val
        pile_sizes[lo_idx] += 1
        pile_tops[lo_idx]   = val

    free(arr)
    free(pile_caps)
    free(pile_tops)

    # --- seed heap: top element of each pile (last appended) ---
    for i in range(num_piles):
        pile_sizes[i] -= 1
        heap_v[heap_size] = piles[i][pile_sizes[i]]
        heap_p[heap_size] = i
        heap_size += 1

    # build min-heap in O(num_piles) via bottom-up heapify
    with nogil:
        i = heap_size // 2 - 1
        while i >= 0:
            heap_sift_down(heap_v, heap_p, heap_size, i)
            i -= 1

        # --- merge ---
        out_idx = 0
        while heap_size > 0:
            pile_idx = heap_p[0]
            top_val  = heap_v[0]

            # pop min from heap: replace root with last element, sift down
            heap_size -= 1
            heap_v[0] = heap_v[heap_size]
            heap_p[0] = heap_p[heap_size]
            if heap_size > 0:
                heap_sift_down(heap_v, heap_p, heap_size, 0)

            output[out_idx] = top_val
            out_idx += 1

            if pile_sizes[pile_idx] > 0:
                pile_sizes[pile_idx] -= 1
                heap_v[heap_size] = piles[pile_idx][pile_sizes[pile_idx]]
                heap_p[heap_size] = pile_idx
                heap_size += 1
                heap_sift_up(heap_v, heap_p, heap_size - 1)

    # --- convert output C array to Python list ---
    cdef list result = [output[i] for i in range(n)]

    # cleanup
    for i in range(num_piles):
        free(piles[i])
    free(piles)
    free(pile_sizes)
    free(heap_v)
    free(heap_p)
    free(output)

    return result
