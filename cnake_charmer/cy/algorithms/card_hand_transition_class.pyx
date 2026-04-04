# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Class-based card/hand transition metrics with deterministic draws (Cython)."""

from cnake_charmer.benchmarks import cython_benchmark

cdef unsigned int MASK32 = 0xFFFFFFFF


cdef class HandMetrics:
    cdef inline int is_pair(self, int av, int bv) noexcept nogil:
        return 1 if av == bv else 0

    cdef inline int is_suited(self, int asuit, int bsuit) noexcept nogil:
        return 1 if asuit == bsuit else 0

    cdef inline int gap(self, int av, int bv) noexcept nogil:
        cdef int g = av - bv
        return -g if g < 0 else g

    cdef inline int score(self, int av, int bv, int asuit, int bsuit) noexcept nogil:
        return av * 17 + bv * 13 + asuit * 7 + bsuit * 5


@cython_benchmark(syntax="cy", args=(140000, 91, 13, 4))
def card_hand_transition_class(int n_hands, int seed, int max_value, int n_suits):
    cdef HandMetrics hm = HandMetrics()
    cdef int i, x, y, av, bv, asuit, bsuit
    cdef int pair_hits = 0
    cdef int suited_hits = 0
    cdef long long gap_sum = 0
    cdef unsigned int score_sum = 0

    with nogil:
        for i in range(n_hands):
            x = (seed * 1103515245 + i * 12345) & 0x7FFFFFFF
            y = (seed * 214013 + i * 2531011 + <int>score_sum) & 0x7FFFFFFF
            av = (x % max_value) + 2
            bv = (y % max_value) + 2
            asuit = x % n_suits
            bsuit = y % n_suits
            pair_hits += hm.is_pair(av, bv)
            suited_hits += hm.is_suited(asuit, bsuit)
            gap_sum += hm.gap(av, bv)
            score_sum = (score_sum + <unsigned int>hm.score(av, bv, asuit, bsuit)) & MASK32

    return (pair_hits, suited_hits, gap_sum, score_sum)
