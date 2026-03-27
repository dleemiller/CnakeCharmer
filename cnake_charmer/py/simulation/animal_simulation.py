"""Simulate predator and prey animals with different movement speeds.

Keywords: simulation, animals, predator, prey, inheritance, movement, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


class Animal:
    """Base animal with position and speed."""

    def __init__(self, x, y, speed):
        self.x = x
        self.y = y
        self.speed = speed
        self.total_distance = 0.0

    def move(self, dx, dy):
        """Move the animal by (dx, dy) scaled by speed."""
        self.x += dx * self.speed
        self.y += dy * self.speed
        dist = (dx * self.speed) ** 2 + (dy * self.speed) ** 2
        self.total_distance += dist**0.5


class Predator(Animal):
    """Predator moves at 2x base speed."""

    def __init__(self, x, y):
        super().__init__(x, y, 2.0)

    def move(self, dx, dy):
        """Predators move faster and have a charge bonus on large moves."""
        mag = (dx * dx + dy * dy) ** 0.5
        bonus = 1.5 if mag > 0.5 else 1.0
        actual_dx = dx * self.speed * bonus
        actual_dy = dy * self.speed * bonus
        self.x += actual_dx
        self.y += actual_dy
        self.total_distance += (actual_dx**2 + actual_dy**2) ** 0.5


class Prey(Animal):
    """Prey moves at 1x base speed."""

    def __init__(self, x, y):
        super().__init__(x, y, 1.0)

    def move(self, dx, dy):
        """Prey moves cautiously: reduced speed for large moves."""
        mag = (dx * dx + dy * dy) ** 0.5
        dampen = 0.7 if mag > 0.5 else 1.0
        actual_dx = dx * self.speed * dampen
        actual_dy = dy * self.speed * dampen
        self.x += actual_dx
        self.y += actual_dy
        self.total_distance += (actual_dx**2 + actual_dy**2) ** 0.5


@python_benchmark(args=(50000,))
def animal_simulation(n: int) -> float:
    """Simulate n animals for 20 steps each, return total distance traveled.

    Args:
        n: Number of animals.

    Returns:
        Total distance traveled by all animals.
    """
    k = 20
    animals = []
    for i in range(n):
        x = ((i * 2654435761 + 17) % 10000) / 100.0
        y = ((i * 1103515245 + 12345) % 10000) / 100.0
        if i % 3 == 0:
            animals.append(Predator(x, y))
        else:
            animals.append(Prey(x, y))

    for step in range(k):
        for i in range(n):
            seed = i * 1664525 + step * 214013 + 1013904223
            dx = ((seed ^ (seed >> 7)) % 1000) / 1000.0 - 0.5
            dy = (
                ((seed * 1103515245 + 12345) ^ ((seed * 1103515245 + 12345) >> 7)) % 1000
            ) / 1000.0 - 0.5
            animals[i].move(dx, dy)

    total = 0.0
    for a in animals:
        total += a.total_distance

    return total
