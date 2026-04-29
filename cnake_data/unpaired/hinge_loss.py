"""Hinge loss and gradient."""

from __future__ import annotations


class Hinge:
    def __init__(self, threshold=1.0):
        self.threshold = float(threshold)

    def set_threshold(self, threshold):
        self.threshold = float(threshold)

    def loss(self, p, y):
        z = p * y
        if z <= self.threshold:
            return self.threshold - z
        return 0.0

    def dloss(self, p, y):
        z = p * y
        if z <= self.threshold:
            return -y
        return 0.0
