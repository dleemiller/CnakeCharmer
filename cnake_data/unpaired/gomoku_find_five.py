"""Five-in-a-row winner checks around last move."""

from __future__ import annotations


def _count_dir(board, size, x, y, dx, dy, nplayer, length):
    c = 0
    for i in range(1, length):
        nx = x + dx * i
        ny = y + dy * i
        if 0 <= nx < size and 0 <= ny < size and board[ny][nx] == nplayer:
            c += 1
        else:
            break
    return c


def find_five(nplayer, board, size, last_move_x, last_move_y):
    x = last_move_x
    y = last_move_y
    length = 5
    if x == -1 or y == -1:
        return 0

    dirs = [(1, 0), (0, 1), (1, -1), (1, 1)]
    for dx, dy in dirs:
        a = _count_dir(board, size, x, y, dx, dy, nplayer, length)
        b = _count_dir(board, size, x, y, -dx, -dy, nplayer, length)
        if a + b + 1 >= length:
            return 1
    return 0
