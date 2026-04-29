"""Simple MA/RSI long-only strategy loop."""

from __future__ import annotations

import math


def run_strategy(
    close: list[float],
    ma_long: list[float],
    ma_short: list[float],
    rsi_n: list[float],
    lower_rsi: float,
    start_index: int,
    posn: list[float],
    cash: list[float],
) -> None:
    long_posn = False
    shares = 0.0

    for idx in range(start_index, len(close)):
        posn[idx] = posn[idx - 1]
        cash[idx] = cash[idx - 1]

        if long_posn and close[idx] > ma_short[idx]:
            long_posn = False
            shares = posn[idx]
            posn[idx] = 0.0
            cash[idx] += shares * close[idx]

        if (not long_posn) and close[idx] > ma_long[idx] and rsi_n[idx] < lower_rsi:
            long_posn = True
            shares = math.floor(cash[idx] / close[idx])
            posn[idx] = shares
            cash[idx] -= shares * close[idx]
