"""Procedural floor generation with adaptive difficulty."""

from __future__ import annotations

from random import Random

SEG_LEN = 20
SEG_AMOUNT = 40
OFFSET_DELTA = 1
MAX_DIFF = 0.25
MIN_DIFF = 0.95
START_Y = 0
SEGMENT_NEURONS = 12


def calc_difficulty(x):
    k = -(((MIN_DIFF - MAX_DIFF) * 100.0) / 1500000.0) * x + MIN_DIFF
    return max(MAX_DIFF, min(MIN_DIFF, k))


class FloorGenerator:
    def __init__(self, starting_diff=None, seed=None):
        self.score = starting_diff if starting_diff is not None else 0
        self.rng = Random(seed) if seed is not None else Random()
        self.offset = 0
        self.floor = [1.0, 1.0, 1.0, 0.0]
        self.current_gap = 0
        self.diff = calc_difficulty(self.score)
        for _ in range(SEG_AMOUNT):
            if self.rng.random() > self.diff and self.current_gap <= 5:
                self.floor.append(0.0)
                self.current_gap += 1
            else:
                self.floor.append(1.0)
                self.current_gap = 0

    def get_segment_neurons(self):
        return self.floor[2 : SEGMENT_NEURONS + 2]

    def next_frame(self):
        if self.offset >= SEG_LEN - OFFSET_DELTA:
            self.score += 1
            self.diff = calc_difficulty(self.score)
            self.floor.pop(0)
            if self.rng.random() > self.diff and self.current_gap <= 5:
                self.floor.append(0.0)
                self.current_gap += 1
            else:
                self.floor.append(1.0)
                self.current_gap = 0
            self.offset = 0
        else:
            self.offset += OFFSET_DELTA

    def agent_died(self, agent_y):
        if agent_y < START_Y:
            return False
        if self.offset < 5:
            return self.floor[2] == 0
        if self.offset >= 15:
            return self.floor[3] == 0
        return self.floor[2] == 0 and self.floor[3] == 0
