# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Class-based Game of Life stepping over a toroidal grid (Cython)."""

from libc.stdlib cimport malloc, free

from cnake_data.benchmarks import cython_benchmark

cdef unsigned int MASK32 = 0xFFFFFFFF


cdef class LifeBoard:
    cdef int width
    cdef int height
    cdef unsigned char *board
    cdef unsigned char *tmp

    def __cinit__(self, int width, int height, int seed):
        cdef int x, y
        self.width = width
        self.height = height
        self.board = <unsigned char *>malloc(width * height * sizeof(unsigned char))
        self.tmp = <unsigned char *>malloc(width * height * sizeof(unsigned char))
        if self.board == NULL or self.tmp == NULL:
            free(self.board)
            free(self.tmp)
            self.board = NULL
            self.tmp = NULL
            raise MemoryError()

        for y in range(height):
            for x in range(width):
                self.board[y * width + x] = 1 if ((seed + x * 17 + y * 31 + x * y) & 7) < 3 else 0

    def __dealloc__(self):
        if self.board != NULL:
            free(self.board)
        if self.tmp != NULL:
            free(self.tmp)

    cdef void step(self) noexcept nogil:
        cdef int w = self.width
        cdef int h = self.height
        cdef int y, x, ym, yp, xm, xp, idx
        cdef int s
        cdef unsigned char alive

        for y in range(h):
            ym = (y - 1 + h) % h
            yp = (y + 1) % h
            for x in range(w):
                xm = (x - 1 + w) % w
                xp = (x + 1) % w
                idx = y * w + x
                s = (
                    self.board[ym * w + xm]
                    + self.board[ym * w + x]
                    + self.board[ym * w + xp]
                    + self.board[y * w + xm]
                    + self.board[y * w + xp]
                    + self.board[yp * w + xm]
                    + self.board[yp * w + x]
                    + self.board[yp * w + xp]
                )
                alive = self.board[idx]
                self.tmp[idx] = 1 if (s == 3 or (alive == 1 and s == 2)) else 0

        for idx in range(w * h):
            self.board[idx] = self.tmp[idx]


@cython_benchmark(syntax="cy", args=(84, 72, 110, 23))
def life_board_steps_class(int width, int height, int steps, int seed):
    cdef LifeBoard life = LifeBoard(width, height, seed)
    cdef int t, x, y
    cdef unsigned int checksum = 0
    cdef int live = 0
    cdef int edge_live = 0

    with nogil:
        for t in range(steps):
            life.step()
            if (t & 7) == 0:
                checksum = (checksum + <unsigned int>(life.board[(t * 13) % (width * height)] * (t + 1))) & MASK32

        for t in range(width * height):
            live += life.board[t]

        for x in range(width):
            edge_live += life.board[x] + life.board[(height - 1) * width + x]
        for y in range(1, height - 1):
            edge_live += life.board[y * width] + life.board[y * width + (width - 1)]

    return (live, edge_live, checksum)
