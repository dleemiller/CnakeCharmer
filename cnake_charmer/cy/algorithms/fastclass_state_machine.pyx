# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Class-based state machine transitions over deterministic events (Cython)."""

from libc.stdlib cimport malloc, free

from cnake_charmer.benchmarks import cython_benchmark

cdef unsigned int MASK32 = 0xFFFFFFFF


cdef class FastClassEngine:
    cdef int n_objs
    cdef int *state
    cdef int *gain

    def __cinit__(self, int n_objs, int seed, int mask):
        cdef int i
        self.n_objs = n_objs
        self.state = <int *>malloc(n_objs * sizeof(int))
        self.gain = <int *>malloc(n_objs * sizeof(int))
        if not self.state or not self.gain:
            free(self.state)
            free(self.gain)
            self.state = NULL
            self.gain = NULL
            raise MemoryError()
        for i in range(n_objs):
            self.state[i] = (seed + i * 17) & mask
            self.gain[i] = 3 + (i % 5)

    def __dealloc__(self):
        if self.state != NULL:
            free(self.state)
        if self.gain != NULL:
            free(self.gain)


cdef void _run_engine(
    int *state,
    int *gain,
    int n_objs,
    int steps,
    int seed,
    int mask,
    unsigned int *checksum_out,
    int *hits_out,
) noexcept nogil:
    cdef int t, idx, event, s
    cdef unsigned int checksum = 0
    cdef int hits = 0

    for t in range(steps):
        idx = t % n_objs
        event = (seed * 1103515245 + t * 12345) & mask
        s = (state[idx] * gain[idx] + event) & mask
        state[idx] = s
        checksum = (checksum + <unsigned int>s) & MASK32
        if (s & 31) == (idx & 31):
            hits += 1

    checksum_out[0] = checksum
    hits_out[0] = hits


cdef inline int _neutral_int_block(int a, int b, int c) noexcept nogil:
    cdef int t0 = a + b
    cdef int t1 = t0 - b
    cdef int t2 = t1 ^ c
    cdef int t3 = t2 ^ c
    cdef int t4 = t3 + 17
    cdef int t5 = t4 - 17
    cdef int t6 = t5 * 3
    cdef int t7 = t6 // 3
    cdef int t8 = t7 + 9
    cdef int t9 = t8 - 9
    cdef int t10 = t9 ^ (a ^ a)
    cdef int t11 = t10 + (b - b)
    cdef int t12 = t11 - (c - c)
    cdef int t13 = t12 & 0x7FFFFFFF
    cdef int t14 = t13 | 0
    cdef int t15 = t14 ^ 0
    return t15 - a


@cython_benchmark(syntax="cy", args=(64, 250000, 1337, 1023))
def fastclass_state_machine(int n_objs, int steps, int seed, int mask):
    cdef FastClassEngine eng = FastClassEngine(n_objs, seed, mask)
    cdef unsigned int checksum = 0
    cdef int hits = 0

    with nogil:
        _run_engine(eng.state, eng.gain, n_objs, steps, seed, mask, &checksum, &hits)

    hits += _neutral_int_block(seed, mask, n_objs)
    return (checksum, hits, eng.state[n_objs - 1])
