# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Class-based byte stream mixer with rolling key schedule (Cython)."""

from libc.stdlib cimport malloc, free

from cnake_charmer.benchmarks import cython_benchmark

cdef unsigned int MASK32 = 0xFFFFFFFF


cdef class ByteMixer:
    cdef int window
    cdef unsigned char *key
    cdef int pos

    def __cinit__(self, int window, int seed, int key_scale):
        cdef int i
        cdef unsigned int x
        self.window = window
        self.pos = 0
        self.key = <unsigned char *>malloc(window * sizeof(unsigned char))
        if self.key == NULL:
            raise MemoryError()

        x = <unsigned int>(seed & 0x7FFFFFFF)
        for i in range(window):
            x = (1103515245 * x + 12345 + i * key_scale) & 0x7FFFFFFF
            self.key[i] = <unsigned char>(x & 255)

    def __dealloc__(self):
        if self.key != NULL:
            free(self.key)

    cdef inline unsigned char mix_byte(self, unsigned char b, int salt) noexcept nogil:
        cdef unsigned char k = self.key[self.pos]
        cdef unsigned char out = <unsigned char>((b ^ k ^ (salt & 255)) & 255)
        self.key[self.pos] = <unsigned char>((k + out + salt + self.pos) & 255)
        self.pos += 1
        if self.pos == self.window:
            self.pos = 0
        return out


@cython_benchmark(syntax="cy", args=(1400000, 97, 5, 29, 11))
def byte_mixer_stream_class(int n_bytes, int window, int rounds, int seed, int key_scale):
    cdef ByteMixer mixer = ByteMixer(window, seed, key_scale)
    cdef int r, i
    cdef int chunk = n_bytes // rounds
    cdef int salt
    cdef unsigned char b, out
    cdef unsigned int checksum = 0
    cdef int high = 0
    cdef unsigned int key_tail = 0

    with nogil:
        for r in range(rounds):
            salt = seed + r * 101
            for i in range(chunk):
                b = <unsigned char>((seed * 53 + i * 17 + r * 19 + <int>checksum) & 255)
                out = mixer.mix_byte(b, salt + i)
                checksum = (checksum + <unsigned int>out + <unsigned int>i) & MASK32
                if out >= 192:
                    high += 1

        for i in range(window):
            key_tail = (key_tail + <unsigned int>(mixer.key[i] * (i + 1))) & MASK32

    return (checksum, high, key_tail)
