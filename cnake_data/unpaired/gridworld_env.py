"""GridWorld transition dynamics for MCTS simulation."""

from __future__ import annotations

UP, DOWN, RIGHT, LEFT = 0, 1, 2, 3


class GridWorld:
    def __init__(self, grid_map, time_limit=-1):
        self.grid_map = grid_map
        self.cur_t = 0
        self.time_limit = time_limit
        lines = grid_map.strip("\n ").split("\n")[4:]
        self.possible_positions = set()
        for x, line in enumerate(lines):
            for y, ch in enumerate(line):
                if ch.upper() == "X":
                    self.cur_x = self.initial_x = x
                    self.cur_y = self.initial_y = y
                if ch.upper() == "G":
                    self.goal_x, self.goal_y = x, y
                if ch in (".", "X", "G"):
                    self.possible_positions.add((x, y))

    def step(self, move):
        self.cur_t += 1
        px, py = self.cur_x, self.cur_y
        if move == UP:
            px -= 1
        elif move == DOWN:
            px += 1
        elif move == RIGHT:
            py += 1
        elif move == LEFT:
            py -= 1

        if (px, py) in self.possible_positions:
            self.cur_x, self.cur_y = px, py

        done = False
        if self.time_limit != -1 and self.cur_t >= self.time_limit:
            done = True
        if self.cur_x == self.goal_x and self.cur_y == self.goal_y:
            done = True
        return (self.cur_x, self.cur_y), -1.0, done

    def reset(self):
        self.cur_x, self.cur_y, self.cur_t = self.initial_x, self.initial_y, 0
        return (self.cur_x, self.cur_y)
