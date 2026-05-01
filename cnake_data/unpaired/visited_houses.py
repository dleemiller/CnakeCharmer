"""Grid-visit counting for alternating walkers."""

from __future__ import annotations


def _move(x, y, movement):
    if movement == "^":
        y += 1
    elif movement == "v":
        y -= 1
    elif movement == ">":
        x += 1
    elif movement == "<":
        x -= 1
    return x, y


def get_visited_houses(path):
    x = y = 0
    houses = {(0, 0)}
    for movement in path:
        x, y = _move(x, y, movement)
        houses.add((x, y))
    return len(houses)


def get_visited_houses_robo(path):
    santa_x = santa_y = 0
    robo_x = robo_y = 0
    houses = {(0, 0)}

    for i, movement in enumerate(path):
        if i % 2 == 0:
            santa_x, santa_y = _move(santa_x, santa_y, movement)
            houses.add((santa_x, santa_y))
        else:
            robo_x, robo_y = _move(robo_x, robo_y, movement)
            houses.add((robo_x, robo_y))
    return len(houses)
