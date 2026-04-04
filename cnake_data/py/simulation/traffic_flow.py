"""Nagel-Schreckenberg traffic flow model (deterministic variant).

Keywords: simulation, traffic, Nagel-Schreckenberg, cellular automaton, flow, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(5000,))
def traffic_flow(n: int) -> tuple:
    """Simulate deterministic Nagel-Schreckenberg traffic flow on a ring road.

    Road has n cells, max speed v_max=5. Initial state: every 3rd cell has a car
    with speed = i % (v_max+1). Instead of random braking, uses deterministic
    braking: car decelerates by 1 if (position + step) % 7 == 0.
    Runs for 500 time steps.

    Rules per step:
    1. Acceleration: v = min(v + 1, v_max)
    2. Slowing: v = min(v, gap - 1) where gap is distance to next car
    3. Deterministic braking: if (pos + step) % 7 == 0: v = max(v - 1, 0)
    4. Move: pos = pos + v

    Args:
        n: Number of road cells.

    Returns:
        Tuple of (total_flow, avg_speed_x1000, num_stopped) where total_flow is
        sum of all speeds over all steps, avg_speed_x1000 is final average speed
        times 1000 as int, and num_stopped is count of zero-speed cars at end.
    """
    v_max = 5
    steps = 500

    # Initialize cars: every 3rd cell
    n_cars = n // 3
    positions = [0] * n_cars
    speeds = [0] * n_cars

    for i in range(n_cars):
        positions[i] = i * 3
        speeds[i] = i % (v_max + 1)

    total_flow = 0

    for step in range(steps):
        # Step 1: Acceleration
        for i in range(n_cars):
            if speeds[i] < v_max:
                speeds[i] += 1

        # Step 2: Slowing (check gap to next car)
        for i in range(n_cars):
            next_car = (i + 1) % n_cars
            gap = positions[next_car] - positions[i]
            if gap <= 0:
                gap += n
            if speeds[i] >= gap:
                speeds[i] = gap - 1
                if speeds[i] < 0:
                    speeds[i] = 0

        # Step 3: Deterministic braking
        for i in range(n_cars):
            if (positions[i] + step) % 7 == 0 and speeds[i] > 0:
                speeds[i] -= 1

        # Step 4: Move and accumulate flow
        for i in range(n_cars):
            positions[i] = (positions[i] + speeds[i]) % n
            total_flow += speeds[i]

    # Final statistics
    speed_sum = 0
    num_stopped = 0
    for i in range(n_cars):
        speed_sum += speeds[i]
        if speeds[i] == 0:
            num_stopped += 1

    avg_speed_x1000 = int(speed_sum * 1000 / n_cars) if n_cars > 0 else 0

    return (total_flow, avg_speed_x1000, num_stopped)
