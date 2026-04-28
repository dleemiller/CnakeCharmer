"""Dyadic segment decomposition utilities."""

from __future__ import annotations

import math


def fit_segments(start_idx: int, end_idx: int, power: int, base: int) -> list[tuple[int, int]]:
    segment_length = int(base**power)
    if end_idx - start_idx < segment_length:
        return []
    start_factor = int(math.ceil(start_idx / segment_length))
    end_factor = int(math.floor(end_idx / segment_length))
    if end_factor == start_factor:
        return []
    return [((fi + 1) * segment_length, power) for fi in range(start_factor, end_factor)]


def _dyadic_breakdown(
    start_idx: int, end_idx: int, base: int, max_power: int
) -> list[tuple[int, int]]:
    if start_idx >= end_idx:
        return []
    if end_idx - start_idx == 1:
        return [(end_idx, 0)]

    cur_power = max_power
    largest_segments: list[tuple[int, int]] = []
    while cur_power >= 0:
        largest_segments = fit_segments(start_idx, end_idx, cur_power, base)
        if largest_segments:
            break
        cur_power -= 1

    if not largest_segments:
        return [(i + 1, 0) for i in range(start_idx, end_idx)]

    left_end = largest_segments[0][0] - base ** largest_segments[0][1]
    return (
        largest_segments
        + _dyadic_breakdown(start_idx, left_end, base, cur_power - 1)
        + _dyadic_breakdown(largest_segments[-1][0], end_idx, base, cur_power - 1)
    )


def dyadic_breakdown(
    start_idx: int, end_idx: int, base: int = 2, max_power: int = -1
) -> list[tuple[int, int]]:
    if end_idx <= start_idx:
        return []
    if max_power < 0:
        max_power = int(math.log(end_idx - start_idx, base))
    return _dyadic_breakdown(start_idx, end_idx, base, max_power)
