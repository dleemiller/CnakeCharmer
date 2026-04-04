# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Class-based collision grid marker updates with block occupancy stats (Cython)."""

from libc.stdlib cimport malloc, free

from cnake_charmer.benchmarks import cython_benchmark

cdef unsigned int MASK32 = 0xFFFFFFFF


cdef class CollisionGrid:
    cdef int width
    cdef int height
    cdef int block
    cdef int nx
    cdef int ny
    cdef int *cells

    def __cinit__(self, int width, int height, int block):
        self.width = width
        self.height = height
        self.block = block
        self.nx = width // block
        self.ny = height // block
        self.cells = <int *>malloc(self.nx * self.ny * sizeof(int))
        if self.cells == NULL:
            raise MemoryError()
        self.clear()

    def __dealloc__(self):
        if self.cells != NULL:
            free(self.cells)

    cdef void clear(self) noexcept nogil:
        cdef int i
        for i in range(self.nx * self.ny):
            self.cells[i] = 0

    cdef inline int _idx(self, int x, int y) noexcept nogil:
        return y * self.nx + x

    cdef void move(self, int old_x, int old_y, int new_x, int new_y) noexcept nogil:
        cdef int oi = self._idx(old_x, old_y)
        cdef int ni = self._idx(new_x, new_y)
        self.cells[oi] -= 1
        self.cells[ni] += 1


@cython_benchmark(syntax="cy", args=(192, 192, 6, 2800, 120, 19))
def collision_grid_markers_class(
    int width,
    int height,
    int block,
    int n_particles,
    int steps,
    int seed,
):
    cdef CollisionGrid grid = CollisionGrid(width, height, block)
    cdef int nx = grid.nx
    cdef int ny = grid.ny
    cdef int *px = <int *>malloc(n_particles * sizeof(int))
    cdef int *py = <int *>malloc(n_particles * sizeof(int))
    cdef int *vx = <int *>malloc(n_particles * sizeof(int))
    cdef int *vy = <int *>malloc(n_particles * sizeof(int))
    cdef int i, t, x, y, ox, oy, nx2, ny2
    cdef unsigned int checksum = 0
    cdef int occupied = 0
    cdef int max_cell = 0
    cdef int v

    if px == NULL or py == NULL or vx == NULL or vy == NULL:
        free(px)
        free(py)
        free(vx)
        free(vy)
        raise MemoryError()

    with nogil:
        for i in range(n_particles):
            x = (seed * 97 + i * 31) % nx
            y = (seed * 53 + i * 29) % ny
            px[i] = x
            py[i] = y
            vx[i] = 1 + ((seed + i) & 1)
            vy[i] = 1 + (((seed >> 1) + i) & 1)
            grid.cells[y * nx + x] += 1

        for t in range(steps):
            for i in range(n_particles):
                ox = px[i]
                oy = py[i]
                nx2 = (ox + vx[i]) % nx
                ny2 = (oy + vy[i]) % ny
                if ((i + t + seed) & 7) == 0:
                    nx2 = (nx2 + 1) % nx
                if ((i + t + seed) & 11) == 0:
                    ny2 = (ny2 + 1) % ny
                grid.move(ox, oy, nx2, ny2)
                px[i] = nx2
                py[i] = ny2
                checksum = (checksum + <unsigned int>(nx2 + 3 * ny2 + grid.cells[ny2 * nx + nx2])) & MASK32

        for i in range(nx * ny):
            v = grid.cells[i]
            if v > 0:
                occupied += 1
            if v > max_cell:
                max_cell = v

    free(px)
    free(py)
    free(vx)
    free(vy)

    return (occupied, max_cell, checksum)
