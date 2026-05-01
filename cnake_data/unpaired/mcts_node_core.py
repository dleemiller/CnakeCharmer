"""Core MCTS node expansion/selection/backprop helpers."""

from __future__ import annotations

import random


class Node:
    def __init__(self, board):
        self.board = board
        self.parent_edge = None
        self.child_edges = {}
        self.expanded = False
        self.is_leaf = board.is_leaf()
        self.edge_count_sum = 0

    def get_highest_ucb_child(self, c):
        best = None
        best_ucb = -float("inf")
        vals = list(self.child_edges.values())
        random.shuffle(vals)
        for edge in vals:
            ucb = edge.ucb(c)
            if ucb > best_ucb:
                best_ucb = ucb
                best = edge
        return best.get_child() if best is not None else None

    def expand(self):
        if self.is_leaf:
            return 0.0
        for option in self.board.get_valid_options():
            board = self.board.copy()
            _, cost, _ = board.play(option)
            child = Node(board)
            edge = self.board.edge_factory(1, self, child, cost)
            child.parent_edge = edge
            self.child_edges[option.opt_id] = edge
        self.edge_count_sum += 1
        self.expanded = True
        score = self.board.get_oracle_score()
        noisy = random.gauss(score, 10.0)
        return min(score, noisy)

    def backprop(self, value):
        cur = self.parent_edge
        while cur is not None:
            cur.update(value)
            cur = cur.get_parent().parent_edge
