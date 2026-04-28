import functools

import numpy as np

MIN_SCORE = 0.001
MAX_SCORE = 9999
NEUTRAL_SCORE = 1
KILL_SCORE = 1.1

YOU_BODY = 1
SNAKE_BODY = 2


def surroundings(head):
    x, y = head
    return ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))


def distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def c_is_dead(board, pos):
    x, y = pos
    if x < 0 or y < 0:
        return True

    width, height = np.shape(board)
    if x >= width or y >= height:
        return True

    val = board[x, y]
    return val in (YOU_BODY, SNAKE_BODY)


@functools.lru_cache(maxsize=128, typed=False)
def is_dead_cached(state_key, pos):
    board, health = state_key
    if health <= 1:
        return True
    return c_is_dead(np.asarray(board), pos)


def is_dead(board_state, pos=None):
    if pos is None:
        pos = board_state.you.head

    if board_state.you.health <= 1:
        return True
    if c_is_dead(board_state.board_array, pos):
        return True

    for s in board_state.other_snakes:
        if pos == s.tail:
            return any(p in board_state.food for p in surroundings(s.head))
        if pos == s.head:
            return len(s.body) >= len(board_state.you.body)
    return False


def score_board_state(board_state):
    if is_dead(board_state):
        return MIN_SCORE

    all_others_dead = True
    num_dead = 0
    for s in board_state.other_snakes:
        bs = board_state.as_snake(s)
        if is_dead(bs):
            num_dead += 1
        else:
            all_others_dead = False

    if all_others_dead:
        return MAX_SCORE

    if all(is_dead(board_state, p) for p in surroundings(board_state.you.head)):
        return MIN_SCORE

    return KILL_SCORE**num_dead if num_dead > 0 else NEUTRAL_SCORE


def minimax_nodes(board_state):
    return [
        board_state.as_snake(board_state.you, with_move=pos)
        for pos in surroundings(board_state.you.head)
        if not c_is_dead(board_state.board_array, pos)
    ]


def combine_snake_scores(scores):
    if not scores:
        return NEUTRAL_SCORE
    min_score = min(scores)
    bonus = sum(0.01 for s in scores if s >= 1.0)
    return min_score + bonus


def minimax_score(board_state, maximizing_player=True, depth=5):
    if maximizing_player:
        board_score = score_board_state(board_state)
        if board_score in (MIN_SCORE, MAX_SCORE) or depth <= 0:
            if board_score == MIN_SCORE:
                return board_score * (9 - depth)
            return board_score

        max_score = MIN_SCORE
        for bs in minimax_nodes(board_state):
            max_score = max(max_score, minimax_score(bs, False, depth - 1))
        return max_score

    scores = []
    new_bs = board_state
    for s in board_state.other_snakes:
        if distance(board_state.you.head, s.head) > depth:
            continue

        snake_scores = [MAX_SCORE]
        min_score = MAX_SCORE

        for bs in minimax_nodes(new_bs.as_snake(s)):
            score = minimax_score(bs.as_snake(board_state.you), True, depth - 1)
            snake_scores.append(score)
            if score < min_score:
                min_score = score
                new_bs = bs

        scores.append(combine_snake_scores(snake_scores))

    if not scores:
        return minimax_score(board_state, True, depth - 1)

    return min(scores)


def apply(board_state, depth=5):
    positions = surroundings(board_state.you.head)
    out = []

    for pos in positions:
        if is_dead(board_state, pos):
            out.append(MIN_SCORE)
        else:
            bs = board_state.as_snake(board_state.you, with_move=pos)
            out.append(minimax_score(bs, False, depth=depth))

    return np.array(out)
