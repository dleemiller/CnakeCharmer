"""Supercover line-of-sight checks on grid opacity map."""

from __future__ import annotations


def get_pos(x, y, grid_height):
    return x * grid_height + y


def calculate_distance(x1, y1, x2, y2):
    return abs(x1 - x2) + abs(y1 - y2)


def get_line(x1, y1, x2, y2, opacity_map, grid_height):
    if x1 == x2 and y1 == y2:
        return True

    dx = x2 - x1
    dy = y2 - y1
    x = x1
    y = y1

    xstep = 1
    ystep = 1
    if dy < 0:
        ystep = -1
        dy = -dy
    if dx < 0:
        xstep = -1
        dx = -dx
    ddy = 2 * dy
    ddx = 2 * dx

    if ddx >= ddy:
        errorprev = error = dx
        for _ in range(dx):
            x += xstep
            error += ddy
            if error > ddx:
                y += ystep
                error -= ddx
                if error + errorprev < ddx:
                    if (x != x2 or y - ystep != y2) and opacity_map[
                        get_pos(x, y - ystep, grid_height)
                    ]:
                        return False
                elif error + errorprev > ddx:
                    if (x - xstep != x2 or y != y2) and opacity_map[
                        get_pos(x - xstep, y, grid_height)
                    ]:
                        return False
                else:
                    if (
                        opacity_map[get_pos(x, y - ystep, grid_height)]
                        and opacity_map[get_pos(x - xstep, y, grid_height)]
                    ):
                        return False
            if (x != x2 or y != y2) and opacity_map[get_pos(x, y, grid_height)]:
                return False
            errorprev = error
    else:
        errorprev = error = dy
        for _ in range(dy):
            y += ystep
            error += ddx
            if error > ddy:
                x += xstep
                error -= ddy
                if error + errorprev < ddy:
                    if (x - xstep != x2 or y != y2) and opacity_map[
                        get_pos(x - xstep, y, grid_height)
                    ]:
                        return False
                elif error + errorprev > ddy:
                    if (x != x2 or y - ystep != y2) and opacity_map[
                        get_pos(x, y - ystep, grid_height)
                    ]:
                        return False
                else:
                    if (
                        opacity_map[get_pos(x, y - ystep, grid_height)]
                        and opacity_map[get_pos(x - xstep, y, grid_height)]
                    ):
                        return False
            if (x != x2 or y != y2) and opacity_map[get_pos(x, y, grid_height)]:
                return False
            errorprev = error

    return x == x2 and y == y2
