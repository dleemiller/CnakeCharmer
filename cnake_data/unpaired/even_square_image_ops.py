"""Helpers for checking and reshaping image stacks to even-square frames."""

from __future__ import annotations


def check_even_square(image_arr: list[list[list[float]]]) -> bool:
    if not image_arr:
        return False
    y = len(image_arr[0])
    x = len(image_arr[0][0]) if y else 0
    return y == x and (x % 2 == 0)


def get_closest_even_square_size(image_arr: list[list[list[float]]]) -> int:
    if not image_arr:
        return 0
    y = len(image_arr[0])
    x = len(image_arr[0][0]) if y else 0
    s = min(y, x)
    if s % 2 == 1:
        s -= 1
    return max(0, s)


def make_even_square(image_arr: list[list[list[float]]]) -> list[list[list[float]]]:
    """Center-crop each frame to closest even square."""
    if not image_arr:
        return []
    size = get_closest_even_square_size(image_arr)
    y = len(image_arr[0])
    x = len(image_arr[0][0]) if y else 0
    y0 = (y - size) // 2
    x0 = (x - size) // 2

    out: list[list[list[float]]] = []
    for frame in image_arr:
        cropped = [row[x0 : x0 + size] for row in frame[y0 : y0 + size]]
        out.append(cropped)
    return out
