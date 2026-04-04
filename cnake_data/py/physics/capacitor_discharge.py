"""Simulate RC circuit capacitor discharge over many time steps.

Keywords: physics, capacitor, discharge, RC circuit, electronics, simulation, benchmark
"""

import math

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(3000,))
def capacitor_discharge(n: int) -> tuple:
    """Simulate n RC circuits discharging in parallel and compute statistics.

    Each circuit i has R = 100 + i*10 ohms, C = 1e-6 * (1 + i*0.01) farads,
    initial voltage V0 = 5.0 + sin(i*0.1) volts.
    Simulates 2000 time steps with dt = 1e-5 seconds using Euler method.
    V(t+dt) = V(t) - V(t) * dt / (R*C).
    Also tracks total energy dissipated: P = V^2/R * dt.

    Args:
        n: Number of RC circuits to simulate.

    Returns:
        Tuple of (sum_final_voltage, total_energy_dissipated, min_final_voltage).
    """
    steps = 2000
    dt = 1e-5

    sum_final_v = 0.0
    total_energy = 0.0
    min_final_v = 1e30

    for i in range(n):
        r = 100.0 + i * 10.0
        c = 1e-6 * (1.0 + i * 0.01)
        rc = r * c
        v = 5.0 + math.sin(i * 0.1)
        energy = 0.0

        for _ in range(steps):
            power_dt = v * v / r * dt
            energy += power_dt
            v -= v * dt / rc

        sum_final_v += v
        total_energy += energy
        if v < min_final_v:
            min_final_v = v

    return (sum_final_v, total_energy, min_final_v)
