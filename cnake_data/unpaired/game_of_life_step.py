from __future__ import annotations


def gol_step(board: list[list[int]]) -> list[list[int]]:
    h = len(board)
    if h == 0:
        return []
    w = len(board[0])
    out = [[0] * w for _ in range(h)]

    for y in range(h):
        ym = (y - 1) % h
        yp = (y + 1) % h
        for x in range(w):
            xm = (x - 1) % w
            xp = (x + 1) % w
            n = (
                board[ym][xm]
                + board[ym][x]
                + board[ym][xp]
                + board[y][xm]
                + board[y][xp]
                + board[yp][xm]
                + board[yp][x]
                + board[yp][xp]
            )
            alive = board[y][x] == 1
            out[y][x] = 1 if (n == 3 or (alive and n == 2)) else 0
    return out
