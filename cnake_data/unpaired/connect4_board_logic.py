"""Core Connect4 board operations and win-state detection."""

from __future__ import annotations


class Board:
    def __init__(self, height: int, width: int, win_length: int):
        self.height = height
        self.width = width
        self.win_length = win_length
        self.pieces = [[0 for _ in range(width)] for _ in range(height)]

    def add_stone(self, column: int, player: int) -> None:
        for r in range(self.height):
            rr = (self.height - 1) - r
            if self.pieces[rr][column] == 0:
                self.pieces[rr][column] = player
                return
        raise ValueError(f"Can't play column {column}")

    def get_valid_moves(self) -> list[int]:
        valid = [0] * self.width
        for c in range(self.width):
            if self.pieces[0][c] == 0:
                valid[c] = 1
        return valid

    def get_win_state(self) -> tuple[bool, int | None]:
        for player in (1, -1):
            for r in range(self.height):
                total = 0
                for c in range(self.width):
                    total = total + 1 if self.pieces[r][c] == player else 0
                    if total == self.win_length:
                        return True, player

            for c in range(self.width):
                total = 0
                for r in range(self.height):
                    total = total + 1 if self.pieces[r][c] == player else 0
                    if total == self.win_length:
                        return True, player

            for r in range(self.height - self.win_length + 1):
                for c in range(self.width - self.win_length + 1):
                    good = True
                    for x in range(self.win_length):
                        if self.pieces[r + x][c + x] != player:
                            good = False
                            break
                    if good:
                        return True, player

                for c in range(self.win_length - 1, self.width):
                    good = True
                    for x in range(self.win_length):
                        if self.pieces[r + x][c - x] != player:
                            good = False
                            break
                    if good:
                        return True, player

        if sum(self.get_valid_moves()) == 0:
            return True, None
        return False, None
