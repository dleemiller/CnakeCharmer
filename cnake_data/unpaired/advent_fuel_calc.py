def advent_fuel_calc(n):
    """Calculate total fuel for n modules (Advent of Code style).

    Each module mass is deterministic. Fuel = mass/3 - 2, recursively
    until fuel <= 0. Returns (total_fuel, max_single_fuel, modules_needing_fuel).
    """
    total = 0
    max_fuel = 0
    modules_with_fuel = 0

    for i in range(n):
        mass = 100 + (i * 37 + 13) % 900
        fuel = 0
        m = mass
        while True:
            f = m // 3 - 2
            if f <= 0:
                break
            fuel += f
            m = f
        total += fuel
        if fuel > max_fuel:
            max_fuel = fuel
        if fuel > 0:
            modules_with_fuel += 1

    return (total, max_fuel, modules_with_fuel)
