"""Aggregate squared errors across smoothing-parameter grids."""

from __future__ import annotations


def sum_squared_errors_i1(
    squared_error: list[list[dict]], alpha: float, gamma: float
) -> dict[tuple[float, float], float]:
    sse = 0.0
    for sq_e in squared_error:
        for row in sq_e:
            if row.get("alpha") == alpha and row.get("gamma") == gamma:
                sse += float(row.get("squared_error", 0.0))
    return {(alpha, gamma): sse}


def sum_squared_errors_i2(
    squared_error: list[list[dict]], smoothing_parameter: float
) -> dict[float, float]:
    sse = 0.0
    for sq_e in squared_error:
        for row in sq_e:
            if row.get("alpha") == smoothing_parameter:
                sse += float(row.get("squared_error", 0.0))
    return {smoothing_parameter: sse}


def sum_squared_errors_i0(squared_error: list[dict]) -> float:
    sse = 0.0
    for row in squared_error:
        sse += float(row.get("squared_error", 0.0))
    return sse
