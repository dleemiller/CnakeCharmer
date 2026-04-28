"""Boundary-layer conductance model."""

from __future__ import annotations

from math import log


def model_conductance(
    von_karman: float,
    height_weather_measurements: float,
    zm: float,
    zh: float,
    d: float,
    plant_height: float,
    wind: float,
) -> float:
    h = max(10.0, plant_height) / 100.0
    top = wind * (von_karman**2)
    den = log((height_weather_measurements - d * h) / (zm * h)) * log(
        (height_weather_measurements - d * h) / (zh * h)
    )
    return top / den
