# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Simulate n particles using nested structs.

Demonstrates nested struct: Vec2 inside Particle.
Simulates gravity toward origin for 50 timesteps.

Keywords: simulation, particle, nested struct, physics, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from libc.math cimport sqrt
from cnake_charmer.benchmarks import cython_benchmark


cdef struct Vec2:
    double x
    double y


cdef struct Particle:
    Vec2 pos
    Vec2 vel
    double mass


@cython_benchmark(syntax="cy", args=(10000,))
def nested_struct_particle(int n):
    """Simulate n particles, return sum of final distances."""
    cdef int i, s
    cdef int steps = 50
    cdef double dt = 0.01
    cdef double dist_sq, force, total_dist
    cdef unsigned int h, h2

    cdef Particle *parts = <Particle *>malloc(
        n * sizeof(Particle)
    )
    if not parts:
        raise MemoryError()

    for i in range(n):
        h = (
            (<unsigned int>i
             * <unsigned int>2654435761)
            ^ (<unsigned int>i
               * <unsigned int>2246822519)
        )
        parts[i].pos.x = (
            <double>(h & 0xFFFF) / 65535.0
            * 10.0 - 5.0
        )
        parts[i].pos.y = (
            <double>((h >> 16) & 0xFFFF)
            / 65535.0 * 10.0 - 5.0
        )
        h2 = (
            (<unsigned int>i
             * <unsigned int>1664525
             + <unsigned int>1013904223)
            ^ (<unsigned int>i
               * <unsigned int>214013)
        )
        parts[i].vel.x = (
            <double>(h2 & 0xFFFF) / 65535.0
            * 2.0 - 1.0
        )
        parts[i].vel.y = (
            <double>((h2 >> 16) & 0xFFFF)
            / 65535.0 * 2.0 - 1.0
        )
        parts[i].mass = 0.5 + (i % 10) * 0.1

    for s in range(steps):
        for i in range(n):
            dist_sq = (
                parts[i].pos.x * parts[i].pos.x
                + parts[i].pos.y * parts[i].pos.y
                + 0.01
            )
            force = -parts[i].mass / dist_sq
            parts[i].vel.x += (
                force * parts[i].pos.x * dt
            )
            parts[i].vel.y += (
                force * parts[i].pos.y * dt
            )
            parts[i].pos.x += parts[i].vel.x * dt
            parts[i].pos.y += parts[i].vel.y * dt

    total_dist = 0.0
    for i in range(n):
        total_dist += sqrt(
            parts[i].pos.x * parts[i].pos.x
            + parts[i].pos.y * parts[i].pos.y
        )

    free(parts)
    return total_dist
