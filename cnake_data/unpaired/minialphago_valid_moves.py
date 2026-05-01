from __future__ import annotations


def valid_moves(board: list[list[int]]) -> list[tuple[int, int]]:
    """Return empty points on a 2D board (0 = empty)."""
    moves: list[tuple[int, int]] = []
    for y, row in enumerate(board):
        for x, v in enumerate(row):
            if v == 0:
                moves.append((y, x))
    return moves


def has_liberty(board: list[list[int]], y: int, x: int) -> bool:
    h = len(board)
    w = len(board[0]) if h else 0
    for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
        if 0 <= ny < h and 0 <= nx < w and board[ny][nx] == 0:
            return True
    return False
