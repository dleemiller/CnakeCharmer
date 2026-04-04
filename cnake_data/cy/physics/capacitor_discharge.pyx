# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Simulate RC circuit capacitor discharge over many time steps (Cython-optimized).

Keywords: physics, capacitor, discharge, RC circuit, electronics, simulation, cython, benchmark
"""

from libc.math cimport sin
from cnake_data.benchmarks import cython_benchmark


@cython_benchmark(syntax="cy", args=(3000,))
def capacitor_discharge(int n):
    """Simulate n RC circuits discharging in parallel and compute statistics."""
    cdef int steps = 2000
    cdef double dt = 1e-5
    cdef double sum_final_v = 0.0
    cdef double total_energy = 0.0
    cdef double min_final_v = 1e30
    cdef int i, s
    cdef double r, c, rc, v, energy, power_dt

    for i in range(n):
        r = 100.0 + i * 10.0
        c = 1e-6 * (1.0 + i * 0.01)
        rc = r * c
        v = 5.0 + sin(i * 0.1)
        energy = 0.0

        for s in range(steps):
            power_dt = v * v / r * dt
            energy += power_dt
            v -= v * dt / rc

        sum_final_v += v
        total_energy += energy
        if v < min_final_v:
            min_final_v = v

    return (sum_final_v, total_energy, min_final_v)
