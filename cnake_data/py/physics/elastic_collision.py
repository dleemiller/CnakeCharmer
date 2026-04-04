"""
Elastic collision momentum transfer simulation.

Two bodies: a small mass near a wall and a large mass approaching it.
Counts total collisions (body-body and wall bounces) using elastic
collision physics. The collision count approximates digits of pi.

Keywords: physics, elastic collision, momentum, simulation, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(5, 1.0))
def elastic_collision(n_digits: int, v_init: float) -> tuple:
    """Simulate elastic collisions between a small and large mass near a wall.

    A small block (mass=1) sits near a wall at x=0. A large block
    (mass=10^(2*(n_digits-1))) approaches from the right with velocity
    -v_init. We count all collisions (block-block and block-wall).

    Args:
        n_digits: Controls mass ratio; m_large = 10^(2*(n_digits-1)).
        v_init: Initial velocity magnitude of the large block (approaches left).

    Returns:
        Tuple of (collision_count, final_small_velocity, final_large_velocity).
    """
    m_small = 1.0
    m_large = 10.0 ** (2 * (n_digits - 1))
    total_mass = m_small + m_large

    # Small block starts at x=1, large block at x=2
    x_small = 1.0
    x_large = 2.0

    # Small block stationary, large block moving left
    v_small = 0.0
    v_large = -v_init

    collision_count = 0

    # Simulation loop: continue while large block moves left
    # or small block can still reach large block
    while True:
        # Time to next body-body collision (if they approach each other)
        t_body = float("inf")
        rel_v = v_small - v_large  # closing speed (positive means approaching)
        gap = x_large - x_small
        if rel_v < 0.0 and gap > 0.0:
            # Bodies moving apart and gap is positive -> no collision
            pass
        elif rel_v > 0.0:
            t_body = gap / rel_v

        # Time to wall collision for small block
        t_wall = float("inf")
        if v_small < 0.0:
            t_wall = -x_small / v_small

        if t_body == float("inf") and t_wall == float("inf"):
            # No more collisions possible
            break

        if t_wall < t_body:
            # Wall collision happens first
            # Advance both blocks to t_wall
            x_small = 0.0  # exactly at wall
            x_large += v_large * t_wall

            # Wall bounce: small block reverses velocity
            v_small = -v_small
            collision_count += 1
        else:
            # Body-body collision
            # Advance both blocks to t_body
            x_small += v_small * t_body
            x_large += v_large * t_body

            # Elastic collision formulas
            new_v_small = ((m_small - m_large) * v_small + 2.0 * m_large * v_large) / total_mass
            new_v_large = ((m_large - m_small) * v_large + 2.0 * m_small * v_small) / total_mass
            v_small = new_v_small
            v_large = new_v_large
            collision_count += 1

        # Check termination: both moving right and large is faster
        if v_small >= 0.0 and v_large >= 0.0 and v_large >= v_small:
            break

    return (collision_count, v_small, v_large)
