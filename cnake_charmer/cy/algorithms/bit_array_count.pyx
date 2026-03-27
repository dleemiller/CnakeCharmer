# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Count set bits using a cdef class bit array with __getitem__, __setitem__, __contains__.

Keywords: bit array, cdef class, __getitem__, __setitem__, __len__, __contains__, benchmark
"""

from libc.stdlib cimport malloc, free
from cnake_charmer.benchmarks import cython_benchmark


cdef class BitArray:
    """Compact bit array with __getitem__, __setitem__, __len__, __contains__."""
    cdef unsigned int *words
    cdef Py_ssize_t nbits
    cdef int num_words

    def __cinit__(self, Py_ssize_t nbits):
        self.nbits = nbits
        self.num_words = (nbits + 31) // 32
        self.words = <unsigned int *>malloc(self.num_words * sizeof(unsigned int))
        if not self.words:
            raise MemoryError()
        cdef int i
        for i in range(self.num_words):
            self.words[i] = 0

    def __dealloc__(self):
        if self.words:
            free(self.words)

    def __len__(self):
        return self.nbits

    def __getitem__(self, Py_ssize_t idx):
        """Return True if bit at idx is set."""
        if idx < 0 or idx >= self.nbits:
            raise IndexError("index out of range")
        return (self.words[idx // 32] >> (idx % 32)) & 1

    def __setitem__(self, Py_ssize_t idx, bint val):
        """Set or clear bit at idx."""
        if idx < 0 or idx >= self.nbits:
            raise IndexError("index out of range")
        cdef int word_idx = idx // 32
        cdef int bit_idx = idx % 32
        if val:
            self.words[word_idx] |= (1u << bit_idx)
        else:
            self.words[word_idx] &= ~(1u << bit_idx)

    def __contains__(self, Py_ssize_t idx):
        """Check if bit at position idx is set (membership test)."""
        if idx < 0 or idx >= self.nbits:
            return False
        return (self.words[idx // 32] >> (idx % 32)) & 1

    cdef int popcount(self):
        """Count total set bits."""
        cdef int count = 0
        cdef unsigned int w
        cdef int i
        for i in range(self.num_words):
            w = self.words[i]
            while w:
                count += w & 1
                w >>= 1
        return count


@cython_benchmark(syntax="cy", args=(100000,))
def bit_array_count(int n):
    """Set bits in a BitArray cdef class and count using popcount + __contains__."""
    cdef BitArray ba = BitArray(n)
    cdef int i
    cdef unsigned int h
    cdef int pos

    # Set bits
    for i in range(n):
        h = ((<unsigned long long>i * <unsigned long long>2654435761 + 13) >> 4) & 0xFF
        if h < 77:
            ba.words[i // 32] |= (1u << (i % 32))

    cdef int pop = ba.popcount()

    # Membership test
    cdef int hits = 0
    for i in range(n // 2):
        pos = ((<unsigned long long>i * <unsigned long long>1103515245 + 7) >> 3) % n
        if (ba.words[pos // 32] >> (pos % 32)) & 1:
            hits += 1

    return pop * 1000 + hits
