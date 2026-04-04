"""Solve multiple Sudoku puzzles using backtracking.

Keywords: algorithms, sudoku, backtracking, constraint satisfaction, benchmark
"""

from cnake_data.benchmarks import python_benchmark


@python_benchmark(args=(80,))
def sudoku_solver(n: int) -> tuple:
    """Solve the same Sudoku puzzle n times using backtracking.

    Uses a medium-difficulty puzzle with ~30 empty cells.

    Args:
        n: Number of times to solve the puzzle.

    Returns:
        Tuple of (solutions_found, first_cell_value, last_cell_value).
    """
    # Medium-difficulty puzzle (0 = empty, 30 empties)
    puzzle = [
        5,
        3,
        0,
        0,
        7,
        0,
        0,
        0,
        0,
        6,
        0,
        0,
        1,
        9,
        5,
        0,
        0,
        0,
        0,
        9,
        8,
        0,
        0,
        0,
        0,
        6,
        0,
        8,
        0,
        0,
        0,
        6,
        0,
        0,
        0,
        3,
        4,
        0,
        0,
        8,
        0,
        3,
        0,
        0,
        1,
        7,
        0,
        0,
        0,
        2,
        0,
        0,
        0,
        6,
        0,
        6,
        0,
        0,
        0,
        0,
        2,
        8,
        0,
        0,
        0,
        0,
        4,
        1,
        9,
        0,
        0,
        5,
        0,
        0,
        0,
        0,
        8,
        0,
        0,
        7,
        9,
    ]

    solutions_found = 0
    first_val = 0
    last_val = 0

    for _ in range(n):
        board = list(puzzle)
        if _solve(board):
            solutions_found += 1
            first_val = board[0]
            last_val = board[80]

    return (solutions_found, first_val, last_val)


def _find_empty(board):
    """Find the next empty cell (value 0). Return index or -1."""
    for i in range(81):
        if board[i] == 0:
            return i
    return -1


def _can_place(board, idx, num):
    """Check if num can be placed at idx without violating constraints."""
    row = idx // 9
    col = idx % 9

    # Check row
    row_start = row * 9
    for c in range(9):
        if board[row_start + c] == num:
            return False

    # Check column
    for r in range(9):
        if board[r * 9 + col] == num:
            return False

    # Check 3x3 box
    box_row = (row // 3) * 3
    box_col = (col // 3) * 3
    for r in range(box_row, box_row + 3):
        for c in range(box_col, box_col + 3):
            if board[r * 9 + c] == num:
                return False

    return True


def _solve(board):
    """Solve the board in-place using backtracking. Return True if solved."""
    idx = _find_empty(board)
    if idx == -1:
        return True

    for num in range(1, 10):
        if _can_place(board, idx, num):
            board[idx] = num
            if _solve(board):
                return True
            board[idx] = 0

    return False
