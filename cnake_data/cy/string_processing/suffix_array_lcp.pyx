# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Build suffix array + LCP array, return sum of LCP values (Cython-optimized).

Keywords: string processing, suffix array, lcp, longest common prefix, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from libc.string cimport memset
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(50000,))
def suffix_array_lcp(int n):
    """Build suffix array + LCP using prefix-doubling SA + Kasai LCP — O(n log n)."""

    cdef int i, gap, a, b
    cdef int *s        # string as ints
    cdef int *sa       # suffix array
    cdef int *rank_arr
    cdef int *tmp_arr  # scratch: second-key order / new ranks
    cdef int *lcp_arr
    cdef int *cnt      # counting-sort buckets
    cdef long long total
    cdef int j, k
    cdef int same
    cdef int ALPHA = 256   # first-pass alphabet size (char values 65-68)

    if n <= 0:
        return 0

    # cnt must hold max(ALPHA, n) entries
    cdef int cnt_size = n if n > ALPHA else ALPHA

    s        = <int *>malloc(n * sizeof(int))
    sa       = <int *>malloc(n * sizeof(int))
    rank_arr = <int *>malloc(n * sizeof(int))
    tmp_arr  = <int *>malloc(n * sizeof(int))
    lcp_arr  = <int *>malloc(n * sizeof(int))
    cnt      = <int *>malloc(cnt_size * sizeof(int))

    if not s or not sa or not rank_arr or not tmp_arr or not lcp_arr or not cnt:
        free(s); free(sa); free(rank_arr); free(tmp_arr); free(lcp_arr); free(cnt)
        raise MemoryError()

    with nogil:
        # ------------------------------------------------------------------ #
        # 1. Build string; initial rank = char value                          #
        # ------------------------------------------------------------------ #
        for i in range(n):
            s[i] = 65 + (i * 7 + 3) % 4   # values in {65,66,67,68}
            rank_arr[i] = s[i]

        # ------------------------------------------------------------------ #
        # 2. Prefix-doubling suffix array — O(n log²n)                       #
        #    Each doubling step: two stable counting sorts.                   #
        # ------------------------------------------------------------------ #

        # Initial SA: sort by single character (values 65-68, use ALPHA buckets)
        memset(cnt, 0, ALPHA * sizeof(int))
        for i in range(n):
            cnt[rank_arr[i]] += 1
        for i in range(1, ALPHA):
            cnt[i] += cnt[i - 1]
        for i in range(n - 1, -1, -1):
            cnt[rank_arr[i]] -= 1
            sa[cnt[rank_arr[i]]] = i

        # Compress initial ranks into [0, n-1] so cnt can always use n buckets
        # after the first re-ranking step
        tmp_arr[sa[0]] = 0
        for i in range(1, n):
            a = sa[i]
            b = sa[i - 1]
            tmp_arr[a] = tmp_arr[b] + (1 if rank_arr[a] != rank_arr[b] else 0)
        for i in range(n):
            rank_arr[i] = tmp_arr[i]

        gap = 1
        while gap < n:
            # ---- Sort by second key (rank of i+gap, sentinel = -1) -------
            # Positions i where i+gap >= n have the smallest second key.
            # Collect them first (in SA order for stability), then the rest.
            j = 0
            for i in range(n - gap, n):
                tmp_arr[j] = i
                j += 1
            for i in range(n):
                if sa[i] >= gap:
                    tmp_arr[j] = sa[i] - gap
                    j += 1
            # tmp_arr[0..n-1] is now sorted by second key (stable)

            # ---- Sort by first key (current rank_arr[i]) — stable ---------
            # After compression rank_arr values are in [0, n-1].
            memset(cnt, 0, n * sizeof(int))
            for i in range(n):
                cnt[rank_arr[tmp_arr[i]]] += 1
            for i in range(1, n):
                cnt[i] += cnt[i - 1]
            for i in range(n - 1, -1, -1):
                a = tmp_arr[i]
                cnt[rank_arr[a]] -= 1
                sa[cnt[rank_arr[a]]] = a

            # ---- Recompute compressed ranks from new SA -------------------
            tmp_arr[sa[0]] = 0
            for i in range(1, n):
                a = sa[i]
                b = sa[i - 1]
                # Same pair (rank[a], rank[a+gap]) vs (rank[b], rank[b+gap])?
                if rank_arr[a] != rank_arr[b]:
                    same = 0
                elif (a + gap < n) != (b + gap < n):
                    same = 0
                elif a + gap < n and rank_arr[a + gap] != rank_arr[b + gap]:
                    same = 0
                else:
                    same = 1
                tmp_arr[a] = tmp_arr[b] + (0 if same else 1)
            for i in range(n):
                rank_arr[i] = tmp_arr[i]

            if rank_arr[sa[n - 1]] == n - 1:
                break   # all ranks unique — SA is complete

            gap *= 2

        # ------------------------------------------------------------------ #
        # 3. Kasai's LCP — O(n)                                              #
        # ------------------------------------------------------------------ #
        # Rebuild position-to-rank mapping from final SA
        for i in range(n):
            rank_arr[sa[i]] = i

        total = 0
        k = 0
        for i in range(n):
            if rank_arr[i] == 0:
                k = 0
                continue
            j = sa[rank_arr[i] - 1]
            while i + k < n and j + k < n and s[i + k] == s[j + k]:
                k += 1
            lcp_arr[rank_arr[i]] = k
            total += k
            if k > 0:
                k -= 1

    free(s); free(sa); free(rank_arr); free(tmp_arr); free(lcp_arr); free(cnt)
    return total
