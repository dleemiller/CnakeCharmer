from dataclasses import dataclass


@dataclass
class AABB:
    center: tuple
    hwidth: float = 0.0
    hheight: float | None = None

    def __post_init__(self):
        if isinstance(self.center, AABB):
            other = self.center
            self.center = tuple(other.center)
            self.hwidth = other.hwidth
            self.hheight = other.hheight
        elif self.hheight is None:
            self.hheight = self.hwidth

    def intersects(self, other):
        return (
            abs(self.center[0] - other.center[0]) * 2 < (self.hwidth + other.hwidth) * 2
            and abs(self.center[1] - other.center[1]) * 2 < (self.hheight + other.hheight) * 2
        )

    def contains_point(self, point):
        return (
            self.center[0] - self.hwidth <= point[0] <= self.center[0] + self.hwidth
            and self.center[1] - self.hheight <= point[1] <= self.center[1] + self.hheight
        )

    def contains_aabb(self, other):
        return (
            self.left() <= other.left()
            and self.right() >= other.right()
            and self.top() <= other.top()
            and self.bottom() >= other.bottom()
        )

    def left(self):
        return self.center[0] - self.hwidth

    def right(self):
        return self.center[0] + self.hwidth

    def top(self):
        return self.center[1] - self.hheight

    def bottom(self):
        return self.center[1] + self.hheight
