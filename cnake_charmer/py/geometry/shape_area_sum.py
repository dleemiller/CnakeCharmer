"""Sum areas of mixed Circle and Rectangle shapes using inheritance.

Keywords: geometry, shapes, area, circle, rectangle, inheritance, benchmark
"""

import math

from cnake_charmer.benchmarks import python_benchmark


class Shape:
    """Base shape with an area method."""

    def area(self):
        return 0.0


class Circle(Shape):
    """Circle shape with radius."""

    def __init__(self, radius):
        self.radius = radius

    def area(self):
        return math.pi * self.radius * self.radius


class Rectangle(Shape):
    """Rectangle shape with width and height."""

    def __init__(self, width, height):
        self.width = width
        self.height = height

    def area(self):
        return self.width * self.height


@python_benchmark(args=(200000,))
def shape_area_sum(n: int) -> float:
    """Create n shapes (circles and rectangles), sum their areas.

    Args:
        n: Number of shapes to create.

    Returns:
        Sum of all shape areas.
    """
    shapes = []
    for i in range(n):
        kind = ((i * 2654435761) >> 4) & 1
        if kind == 0:
            radius = ((i * 1664525 + 1013904223) % 10000) / 100.0
            shapes.append(Circle(radius))
        else:
            width = ((i * 1103515245 + 12345) % 10000) / 100.0
            height = ((i * 214013 + 2531011) % 10000) / 100.0
            shapes.append(Rectangle(width, height))

    total = 0.0
    for s in shapes:
        total += s.area()

    return total
