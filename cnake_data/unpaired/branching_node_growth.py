"""Geometric branching node growth model."""

from __future__ import annotations

import random
from math import cos, sin, sqrt


class Node:
    def __init__(
        self,
        x,
        y,
        radius,
        angle,
        split_rate,
        branch_rate,
        single_child_rad_rat,
        split_child_rad_rat,
        branch_child_rad_rat,
        single_child_angle_sigma,
        split_child_angle_dev,
        branch_child_angle_dev,
        child_split_k,
        child_branch_k,
    ):
        self.x = x
        self.y = y
        self.radius = radius
        self.norm = sqrt(self.x * self.x + self.y * self.y)
        self.angle = angle
        self._split_rate = split_rate
        self._branch_rate = branch_rate
        self._single_crr = single_child_rad_rat
        self._split_crr = split_child_rad_rat
        self._branch_crr = branch_child_rad_rat
        self._single_cas = single_child_angle_sigma
        self._split_cad = split_child_angle_dev
        self._branch_cad = branch_child_angle_dev
        self._child_sk = child_split_k
        self._child_bk = child_branch_k
        if split_rate + branch_rate > 1:
            s = split_rate + branch_rate
            self._split_rate = split_rate / s
            self._branch_rate = branch_rate / s

    def spawn(self):
        r_split = random.random()
        r_branch_l = random.random()
        r_branch_r = random.random()
        children = []
        if r_split < self._split_rate:
            children.extend(self._build_split())
        else:
            children.append(self._build_single())
            if r_branch_l < self._branch_rate:
                children.append(self._build_left_branch())
            if r_branch_r < self._branch_rate:
                children.append(self._build_right_branch())
        return children

    def _build_split(self):
        radius = self.radius * self._split_crr
        return [
            self._build_child(radius, self.angle + self._split_cad),
            self._build_child(radius, self.angle - self._split_cad),
        ]

    def _build_left_branch(self):
        return self._build_child(self.radius * self._branch_crr, self.angle + self._branch_cad)

    def _build_right_branch(self):
        return self._build_child(self.radius * self._branch_crr, self.angle - self._branch_cad)

    def _build_single(self):
        radius = self.radius * self._single_crr
        angle = self.angle + random.gauss(0, self._single_cas)
        return self._build_child(radius, angle)

    def _build_child(self, radius, angle):
        x = self.x + (self.radius + radius) * cos(angle)
        y = self.y + (self.radius + radius) * sin(angle)
        return Node(
            x,
            y,
            radius,
            angle,
            self._split_rate * self._child_sk,
            self._branch_rate * self._child_bk,
            self._single_crr,
            self._split_crr,
            self._branch_crr,
            self._single_cas,
            self._split_cad,
            self._branch_cad,
            self._child_sk,
            self._child_bk,
        )

    def distance(self, other):
        dx = self.x - other.x
        dy = self.y - other.y
        return sqrt(dx * dx + dy * dy)

    def intersects(self, other):
        return 1 if self.distance(other) <= self.radius + other.radius - 1 else 0
