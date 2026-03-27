# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Compute cross products of 3D vector pairs and sum the results using a cdef class.

Keywords: vector, cross product, 3D, geometry, cdef class, __add__, __sub__, __neg__, cython, benchmark
"""

from cnake_charmer.benchmarks import cython_benchmark


cdef class Vec3D:
    cdef double x
    cdef double y
    cdef double z

    def __init__(self, double x, double y, double z):
        self.x = x
        self.y = y
        self.z = z

    def __add__(x, y):
        cdef Vec3D a = <Vec3D>x
        cdef Vec3D b = <Vec3D>y
        return Vec3D(a.x + b.x, a.y + b.y, a.z + b.z)

    def __sub__(x, y):
        cdef Vec3D a = <Vec3D>x
        cdef Vec3D b = <Vec3D>y
        return Vec3D(a.x - b.x, a.y - b.y, a.z - b.z)

    def __neg__(self):
        return Vec3D(-self.x, -self.y, -self.z)

    cdef Vec3D cross(self, Vec3D other):
        return Vec3D(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )


@cython_benchmark(syntax="cy", args=(100000,))
def vector3d_cross_sum(int n):
    """Generate n 3D vector pairs, compute cross products, and sum.

    Uses operator overloading (__add__, __sub__, __neg__) to accumulate
    cross product results. Returns (x, y, z) of summed vector.

    Args:
        n: Number of vector pairs.

    Returns:
        Tuple of (x, y, z) sums.
    """
    cdef Vec3D total = Vec3D(0.0, 0.0, 0.0)
    cdef Vec3D a, b, c, neg_c
    cdef int i
    cdef unsigned long long h1, h2, h3, h4, h5, h6
    cdef double ax, ay, az, bx, by, bz

    for i in range(n):
        h1 = ((<unsigned long long>i * <unsigned long long>2654435761 + 1) >> 8) & 0xFFFF
        h2 = ((<unsigned long long>i * <unsigned long long>1103515245 + 3) >> 8) & 0xFFFF
        h3 = ((<unsigned long long>i * <unsigned long long>2246822519 + 5) >> 8) & 0xFFFF
        ax = (<int>(h1 % 201) - 100) / 10.0
        ay = (<int>(h2 % 201) - 100) / 10.0
        az = (<int>(h3 % 201) - 100) / 10.0

        h4 = ((<unsigned long long>i * <unsigned long long>6364136223846793005 + 7) >> 16) & 0xFFFF
        h5 = ((<unsigned long long>i * <unsigned long long>3935559000370003845 + 11) >> 16) & 0xFFFF
        h6 = ((<unsigned long long>i * <unsigned long long>2862933555777941757 + 13) >> 16) & 0xFFFF
        bx = (<int>(h4 % 201) - 100) / 10.0
        by = (<int>(h5 % 201) - 100) / 10.0
        bz = (<int>(h6 % 201) - 100) / 10.0

        a = Vec3D(ax, ay, az)
        b = Vec3D(bx, by, bz)

        # Cross product using operator overloading for accumulation
        c = a.cross(b)
        # Use __neg__ and __sub__ to demonstrate operators:
        # total = total + c  is same as  total = total - (-c)
        neg_c = -c
        total = total - neg_c

    return (total.x, total.y, total.z)
