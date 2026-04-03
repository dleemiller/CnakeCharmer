import math


def net_radiation(n):
    """Compute net radiation equivalent evaporation for n timesteps.

    Simple energy balance model. Returns (total_evap, max_evap, mean_evap).
    """
    lambda_v = 2.45  # MJ/kg
    total = 0.0
    max_evap = 0.0

    for i in range(n):
        # Deterministic net radiation (sinusoidal daily pattern)
        net_rad = 15.0 * math.sin(math.pi * (i % 24) / 24.0) + 2.0
        if net_rad < 0:
            net_rad = 0.0
        evap = net_rad / lambda_v
        total += evap
        if evap > max_evap:
            max_evap = evap

    mean_evap = total / n if n > 0 else 0.0
    return (round(total, 4), round(max_evap, 4), round(mean_evap, 4))
