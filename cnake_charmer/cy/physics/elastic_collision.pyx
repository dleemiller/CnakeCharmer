# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""
Elastic collision momentum transfer simulation (Cython-optimized).

Two bodies: a small mass near a wall and a large mass approaching it.
Counts total collisions using cdef helper functions for velocity transfer
and collision checking.

Keywords: physics, elastic collision, momentum, simulation, cython, benchmark
"""

from libc.math cimport pow
from cnake_charmer.benchmarks import cython_benchmark


cdef double INFINITY = 1e308


cdef (double, double) momentum_transfer(double m_small, double m_large,
                                         double total_mass,
                                         double v_small, double v_large) nogil:
    """Compute post-collision velocities for elastic collision."""
    cdef double new_v_small, new_v_large
    new_v_small = ((m_small - m_large) * v_small + 2.0 * m_large * v_large) / total_mass
    new_v_large = ((m_large - m_small) * v_large + 2.0 * m_small * v_small) / total_mass
    return new_v_small, new_v_large


cdef double time_to_body_collision(double x_small, double x_large,
                                    double v_small, double v_large) nogil:
    """Compute time until the two bodies collide, or INFINITY if they won't."""
    cdef double rel_v, gap
    rel_v = v_small - v_large
    gap = x_large - x_small
    if rel_v < 0.0 and gap > 0.0:
        return INFINITY
    if rel_v > 0.0:
        return gap / rel_v
    return INFINITY


cdef double time_to_wall_collision(double x_small, double v_small) nogil:
    """Compute time until the small block hits the wall at x=0."""
    if v_small < 0.0:
        return -x_small / v_small
    return INFINITY


@cython_benchmark(syntax="cy", args=(5, 1.0))
def elastic_collision(int n_digits, double v_init):
    """Simulate elastic collisions between a small and large mass near a wall.

    Args:
        n_digits: Controls mass ratio; m_large = 10^(2*(n_digits-1)).
        v_init: Initial velocity magnitude of the large block.

    Returns:
        Tuple of (collision_count, final_small_velocity, final_large_velocity).
    """
    cdef double m_small = 1.0
    cdef double m_large = pow(10.0, 2.0 * (n_digits - 1))
    cdef double total_mass = m_small + m_large
    cdef double x_small = 1.0
    cdef double x_large = 2.0
    cdef double v_small = 0.0
    cdef double v_large = -v_init
    cdef int collision_count = 0
    cdef double t_body, t_wall
    cdef double new_v_small, new_v_large

    while True:
        t_body = time_to_body_collision(x_small, x_large, v_small, v_large)
        t_wall = time_to_wall_collision(x_small, v_small)

        if t_body == INFINITY and t_wall == INFINITY:
            break

        if t_wall < t_body:
            # Wall collision
            x_large = x_large + v_large * t_wall
            x_small = 0.0
            v_small = -v_small
            collision_count += 1
        else:
            # Body-body collision
            x_small = x_small + v_small * t_body
            x_large = x_large + v_large * t_body
            new_v_small, new_v_large = momentum_transfer(
                m_small, m_large, total_mass, v_small, v_large)
            v_small = new_v_small
            v_large = new_v_large
            collision_count += 1

        # Both moving right, large faster -> no more collisions
        if v_small >= 0.0 and v_large >= 0.0 and v_large >= v_small:
            break

    return (collision_count, v_small, v_large)
