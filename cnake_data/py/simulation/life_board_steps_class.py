"""Class-based Game of Life stepping over a toroidal grid.

Keywords: simulation, class, game of life, cellular automata, benchmark
"""

from cnake_data.benchmarks import python_benchmark


class LifeBoard:
    def __init__(self, width: int, height: int, seed: int):
        self.width = width
        self.height = height
        self.board = [0] * (width * height)
        for y in range(height):
            for x in range(width):
                self.board[y * width + x] = 1 if ((seed + x * 17 + y * 31 + x * y) & 7) < 3 else 0

    def step(self) -> None:
        w = self.width
        h = self.height
        cur = self.board
        nxt = [0] * (w * h)
        for y in range(h):
            ym = (y - 1 + h) % h
            yp = (y + 1) % h
            for x in range(w):
                xm = (x - 1 + w) % w
                xp = (x + 1) % w
                idx = y * w + x
                s = (
                    cur[ym * w + xm]
                    + cur[ym * w + x]
                    + cur[ym * w + xp]
                    + cur[y * w + xm]
                    + cur[y * w + xp]
                    + cur[yp * w + xm]
                    + cur[yp * w + x]
                    + cur[yp * w + xp]
                )
                alive = cur[idx]
                nxt[idx] = 1 if (s == 3 or (alive == 1 and s == 2)) else 0
        self.board = nxt


@python_benchmark(args=(84, 72, 110, 23))
def life_board_steps_class(width: int, height: int, steps: int, seed: int) -> tuple:
    life = LifeBoard(width, height, seed)
    checksum = 0

    for t in range(steps):
        life.step()
        if (t & 7) == 0:
            checksum = (checksum + life.board[(t * 13) % (width * height)] * (t + 1)) & 0xFFFFFFFF

    live = sum(life.board)
    edge_live = 0
    for x in range(width):
        edge_live += life.board[x] + life.board[(height - 1) * width + x]
    for y in range(1, height - 1):
        edge_live += life.board[y * width] + life.board[y * width + (width - 1)]

    return (live, edge_live, checksum)
