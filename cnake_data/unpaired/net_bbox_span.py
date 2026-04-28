class Net:
    """Simple net with pin set and Manhattan bbox span cost."""

    def __init__(self, net_id):
        self.id = int(net_id)
        self.pins = set()
        self.cost = 0

    def update_ltrb(self):
        """Update cached bounding-box span cost from current pins."""
        if not self.pins:
            self.cost = 0
            return self.cost

        pin_iter = iter(self.pins)
        x0, y0 = next(pin_iter)
        left = right = x0
        top = bottom = y0

        for x, y in pin_iter:
            if x < left:
                left = x
            elif x > right:
                right = x

            if y < top:
                top = y
            elif y > bottom:
                bottom = y

        self.cost = (right - left) + (bottom - top)
        return self.cost

    def __repr__(self):
        return f"<Net id={self.id}>"
