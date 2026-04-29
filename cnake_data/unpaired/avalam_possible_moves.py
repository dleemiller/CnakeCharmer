"""Enumerate legal Avalam-like tower moves on a 9x9 board."""

from __future__ import annotations


def possible_moves(board):
    moves = []
    for a in range(9):
        for b in range(9):
            tower = board[a][b]
            if tower == []:
                continue
            for c in range(-1, 2):
                for d in range(-1, 2):
                    na, nb = a + c, b + d
                    if not (0 <= na < 9 and 0 <= nb < 9):
                        continue
                    other = board[na][nb]
                    if other == [] or (c == 0 and d == 0):
                        continue
                    if len(tower) + len(other) <= 5:
                        moves.append([[a, b], [na, nb], len(tower)])
    return moves
