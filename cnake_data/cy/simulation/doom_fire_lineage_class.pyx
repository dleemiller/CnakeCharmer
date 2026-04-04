# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Class-based doom-fire style heat propagation simulation (Cython)."""

from libc.stdlib cimport malloc, free

from cnake_data.benchmarks import cython_benchmark

cdef unsigned int MASK32 = 0xFFFFFFFF


cdef class DoomFire:
    cdef int width
    cdef int height
    cdef unsigned char *heat
    cdef unsigned char *tmp

    def __cinit__(self, int width, int height):
        cdef int i
        self.width = width
        self.height = height
        self.heat = <unsigned char *>malloc(width * height * sizeof(unsigned char))
        self.tmp = <unsigned char *>malloc(width * height * sizeof(unsigned char))
        if self.heat == NULL or self.tmp == NULL:
            free(self.heat)
            free(self.tmp)
            self.heat = NULL
            self.tmp = NULL
            raise MemoryError()
        for i in range(width * height):
            self.heat[i] = 0
            self.tmp[i] = 0

    def __dealloc__(self):
        if self.heat != NULL:
            free(self.heat)
        if self.tmp != NULL:
            free(self.tmp)

    cdef void ignite_bottom(self, int base) noexcept nogil:
        cdef int x
        cdef int off = (self.height - 1) * self.width
        for x in range(self.width):
            self.heat[off + x] = <unsigned char>((base + x * 3) & 255)

    cdef void spread_step(self, int cooling, int jitter) noexcept nogil:
        cdef int w = self.width
        cdef int h = self.height
        cdef int y, x, row, up, decay, v, nx, i

        for i in range(w * h):
            self.tmp[i] = self.heat[i]

        for y in range(1, h):
            row = y * w
            up = (y - 1) * w
            for x in range(w):
                decay = cooling + ((x + y + jitter) & 3)
                v = <int>self.heat[row + x] - decay
                if v < 0:
                    v = 0
                nx = x - (jitter & 1)
                if nx < 0:
                    nx += w
                self.tmp[up + nx] = <unsigned char>v

        for i in range(w * h):
            self.heat[i] = self.tmp[i]


@cython_benchmark(syntax="cy", args=(90, 70, 140, 3, 37))
def doom_fire_lineage_class(int width, int height, int steps, int cooling, int seed):
    cdef DoomFire fire = DoomFire(width, height)
    cdef int t, i
    cdef unsigned int checksum = 0
    cdef int total = 0
    cdef int nonzero = 0
    cdef int peak = 0
    cdef int v

    with nogil:
        fire.ignite_bottom((seed * 29) & 255)
        for t in range(steps):
            fire.spread_step(cooling, seed + t)
            if (t & 15) == 0:
                checksum = (checksum + <unsigned int>fire.heat[(t * 7) % (width * height)]) & MASK32

        for i in range(width * height):
            v = fire.heat[i]
            total += v
            if v > 0:
                nonzero += 1
            if v > peak:
                peak = v

    return (total, nonzero, peak, checksum)
