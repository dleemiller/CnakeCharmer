# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Class-based particle motion with drag and path accumulation (Cython)."""

from libc.stdlib cimport malloc, free

from cnake_charmer.benchmarks import cython_benchmark


cdef class DragSystem:
    cdef int n
    cdef double *x
    cdef double *y
    cdef double *vx
    cdef double *vy

    def __cinit__(self, int n):
        cdef int i
        self.n = n
        self.x = <double *>malloc(n * sizeof(double))
        self.y = <double *>malloc(n * sizeof(double))
        self.vx = <double *>malloc(n * sizeof(double))
        self.vy = <double *>malloc(n * sizeof(double))
        if not self.x or not self.y or not self.vx or not self.vy:
            free(self.x)
            free(self.y)
            free(self.vx)
            free(self.vy)
            self.x = NULL
            self.y = NULL
            self.vx = NULL
            self.vy = NULL
            raise MemoryError()
        for i in range(n):
            self.x[i] = i * 0.02
            self.y[i] = -i * 0.01
            self.vx[i] = (i % 7) * 0.04
            self.vy[i] = (i % 5) * -0.03

    def __dealloc__(self):
        if self.x != NULL:
            free(self.x)
        if self.y != NULL:
            free(self.y)
        if self.vx != NULL:
            free(self.vx)
        if self.vy != NULL:
            free(self.vy)


cdef double _advance_drag(double *x, double *y, double *vx, double *vy, int n, int steps, double dt, double drag) noexcept nogil:
    cdef int t, i
    cdef double ax, ay
    cdef double path_sum = 0.0
    for t in range(steps):
        for i in range(n):
            ax = ((t + i * 7) % 13 - 6) * 0.01
            ay = ((t + i * 5) % 11 - 5) * 0.01
            vx[i] = (vx[i] + ax * dt) * (1.0 - drag)
            vy[i] = (vy[i] + ay * dt) * (1.0 - drag)
            x[i] += vx[i] * dt
            y[i] += vy[i] * dt
            path_sum += x[i] * 0.3 + y[i] * 0.7
    return path_sum


@cython_benchmark(syntax="cy", args=(140, 1800, 0.008, 0.03))
def particle_drag_path_class(int n, int steps, double dt, double drag):
    cdef DragSystem sys = DragSystem(n)
    cdef double path_sum
    with nogil:
        path_sum = _advance_drag(sys.x, sys.y, sys.vx, sys.vy, n, steps, dt, drag)
    return (path_sum, sys.x[n // 2], sys.y[n // 2])
