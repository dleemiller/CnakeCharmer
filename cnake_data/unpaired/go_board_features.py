"""Feature extraction helpers for Go board states."""

from __future__ import annotations


def label_from_move(move, size):
    label = [0] * (size * size)
    label[move[0] + move[1] * size] = 1
    return label


def winner_label(game_result, blacks_move):
    if game_result.startswith("W"):
        return 1 if not blacks_move else -1
    if game_result.startswith("B"):
        return 1 if blacks_move else -1
    return None


def input_from_board(stones, freedoms, size, is_blacks_turn):
    white = [1 if s == 1 else 0 for s in stones]
    black = [1 if s == -1 else 0 for s in stones]
    if not is_blacks_turn:
        black, white = white, black

    inp = []
    for y in range(size):
        for x in range(size):
            i = x + y * size
            inp.append(black[i])
            inp.append(white[i])
            inp.append(min(freedoms[i], 4))
            inp.append(x / (size - 1) - 0.5)
            inp.append(y / (size - 1) - 0.5)
    return inp
