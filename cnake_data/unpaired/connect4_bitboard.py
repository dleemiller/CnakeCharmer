"""Bitboard-style Connect4 board utilities."""

from __future__ import annotations

import numpy as np

BH = 6
BW = 7
MOVES = 42
BLACK = 1
WHITE = -1
EMPTY = 0


def popcnt(n: int) -> int:
    return int(n.bit_count())


class Board:
    def __init__(self):
        self.black = 0
        self.white = 0

    def copy(self) -> Board:
        out = Board()
        out.black = self.black
        out.white = self.white
        return out

    def reversed_copy(self) -> Board:
        out = Board()
        out.black = self.white
        out.white = self.black
        return out

    def get_value(self, y: int, x: int) -> int:
        i = y * BW + x
        bit = 1 << i
        if self.black & bit:
            return BLACK
        if self.white & bit:
            return WHITE
        return EMPTY

    def set_value(self, y: int, x: int, v: int) -> None:
        i = y * BW + x
        bit = 1 << i
        if v == BLACK:
            self.black |= bit
            self.white &= ~bit
        elif v == WHITE:
            self.white |= bit
            self.black &= ~bit
        else:
            self.black &= ~bit
            self.white &= ~bit

    def to_array1d(self) -> np.ndarray:
        ary = np.zeros(MOVES, dtype=int)
        for i in range(MOVES):
            bit = 1 << i
            if self.black & bit:
                ary[i] = BLACK
            elif self.white & bit:
                ary[i] = WHITE
        return ary

    def legal_moves(self) -> np.ndarray:
        ret = np.zeros(MOVES, dtype=int)
        occupied = self.black | self.white
        for x in range(BW):
            for y in range(BH - 1, -1, -1):
                i = y * BW + x
                if (occupied & (1 << i)) == 0:
                    ret[i] = 1
                    break
        return ret


def create_board_from_array2d(cell_ary: np.ndarray) -> Board:
    out = Board()
    for y in range(BH):
        for x in range(BW):
            out.set_value(y, x, int(cell_ary[y, x]))
    return out
