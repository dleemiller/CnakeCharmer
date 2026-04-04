# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Solve multiple Sudoku puzzles using backtracking (Cython-optimized).

Keywords: algorithms, sudoku, backtracking, constraint satisfaction, cython, benchmark
"""

from libc.stdlib cimport malloc, free
from libc.string cimport memcpy
from cnake_data.benchmarks import cython_benchmark


cdef bint can_place(int *board, int idx, int num) nogil:
    """Check if num can be placed at idx without violating constraints."""
    cdef int row = idx / 9
    cdef int col = idx % 9
    cdef int row_start = row * 9
    cdef int c, r
    cdef int box_row, box_col

    # Check row
    for c in range(9):
        if board[row_start + c] == num:
            return False

    # Check column
    for r in range(9):
        if board[r * 9 + col] == num:
            return False

    # Check 3x3 box
    box_row = (row / 3) * 3
    box_col = (col / 3) * 3
    for r in range(box_row, box_row + 3):
        for c in range(box_col, box_col + 3):
            if board[r * 9 + c] == num:
                return False

    return True


cdef int find_empty(int *board) nogil:
    """Find the next empty cell. Return index or -1."""
    cdef int i
    for i in range(81):
        if board[i] == 0:
            return i
    return -1


cdef bint solve(int *board) nogil:
    """Solve the board in-place using backtracking."""
    cdef int idx = find_empty(board)
    cdef int num
    if idx == -1:
        return True

    for num in range(1, 10):
        if can_place(board, idx, num):
            board[idx] = num
            if solve(board):
                return True
            board[idx] = 0

    return False


@cython_benchmark(syntax="cy", args=(80,))
def sudoku_solver(int n):
    """Solve the same Sudoku puzzle n times using backtracking with C arrays."""
    # Medium-difficulty puzzle (0 = empty, 30 empties)
    cdef int template[81]
    template[:] = [
        5, 3, 0, 0, 7, 0, 0, 0, 0,
        6, 0, 0, 1, 9, 5, 0, 0, 0,
        0, 9, 8, 0, 0, 0, 0, 6, 0,
        8, 0, 0, 0, 6, 0, 0, 0, 3,
        4, 0, 0, 8, 0, 3, 0, 0, 1,
        7, 0, 0, 0, 2, 0, 0, 0, 6,
        0, 6, 0, 0, 0, 0, 2, 8, 0,
        0, 0, 0, 4, 1, 9, 0, 0, 5,
        0, 0, 0, 0, 8, 0, 0, 7, 9,
    ]

    cdef int *board = <int *>malloc(81 * sizeof(int))
    if not board:
        raise MemoryError()

    cdef int solutions_found = 0
    cdef int first_val = 0
    cdef int last_val = 0
    cdef int i

    for i in range(n):
        memcpy(board, template, 81 * sizeof(int))
        if solve(board):
            solutions_found += 1
            first_val = board[0]
            last_val = board[80]

    free(board)
    return (solutions_found, first_val, last_val)
