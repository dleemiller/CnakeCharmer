"""Permutation-based 2x2 cube state operations."""

from __future__ import annotations

import numpy as np

UP = ((0, 2, 7, 5), (1, 4, 6, 3), (8, 47, 14, 11), (9, 46, 15, 12), (10, 45, 16, 13))
LEFT = ((8, 10, 25, 23), (9, 18, 24, 17), (0, 11, 32, 40), (3, 19, 35, 43), (5, 26, 37, 45))
FRONT = ((11, 13, 28, 26), (12, 20, 27, 19), (5, 14, 34, 25), (6, 21, 33, 18), (7, 29, 32, 10))
RIGHT = ((14, 16, 31, 29), (15, 22, 30, 21), (2, 42, 34, 13), (4, 44, 36, 20), (7, 47, 39, 28))
DOWN = ((32, 34, 39, 37), (33, 36, 38, 35), (23, 26, 29, 42), (24, 27, 30, 41), (25, 29, 31, 40))
BOTTOM = ((40, 42, 47, 45), (41, 44, 46, 43), (0, 23, 39, 16), (1, 17, 38, 22), (8, 37, 31, 2))


def _reverse_perm(face):
    return tuple(tuple(reversed(c)) for c in reversed(face))


MOVE_MAP = {
    "U": UP,
    "L": LEFT,
    "F": FRONT,
    "R": RIGHT,
    "D": DOWN,
    "B": BOTTOM,
    "U'": _reverse_perm(UP),
    "L'": _reverse_perm(LEFT),
    "F'": _reverse_perm(FRONT),
    "R'": _reverse_perm(RIGHT),
    "D'": _reverse_perm(DOWN),
    "B'": _reverse_perm(BOTTOM),
}


class Cube48:
    def __init__(self):
        self.v = np.arange(1, 49)
        self.came_from = []
        self.hash = hash(self.v.tobytes())

    def copy(self) -> Cube48:
        c = Cube48()
        c.v = np.copy(self.v)
        c.came_from = self.came_from[:]
        c.hash = self.hash
        return c

    def _rehash(self):
        self.hash = hash(self.v.tobytes())

    def permute(self, seq):
        last_value = self.v[seq[-1]]
        for i in range(len(seq) - 1, 0, -1):
            self.v[seq[i]] = self.v[seq[i - 1]]
        self.v[seq[0]] = last_value

    def apply_permutations(self, permutations):
        for p in permutations:
            self.permute(p)

    def apply_moves(self, moves):
        for move in moves:
            self.apply_permutations(MOVE_MAP[move])
        self._rehash()

    @staticmethod
    def parse_moves(moves: str):
        parts = [c for c in moves.split(" ") if c]
        out = []
        for move in parts:
            if len(move) == 1 and move in MOVE_MAP:
                out.append(move)
            elif len(move) == 2 and move[0] in MOVE_MAP:
                if move[1] == "2":
                    out.extend([move[0], move[0]])
                elif move[1] == "'":
                    out.append(move)
                else:
                    raise ValueError("Invalid move modifier", move)
            else:
                raise ValueError("Invalid move", move)
        return out

    def is_solved(self):
        return np.all(self.v == np.arange(1, 49))


def h_try(candidate: Cube48, solution: Cube48, mask) -> int:
    accum = 0
    for i in range(len(mask)):
        accum += int(candidate.v[mask[i]] != solution.v[mask[i]])
    return accum
