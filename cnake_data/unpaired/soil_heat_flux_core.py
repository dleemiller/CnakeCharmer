from __future__ import annotations


def soil_heat_flux(temp: list[float], k: float, dz: float) -> list[float]:
    """Compute flux between adjacent layers from 1D temperature profile."""
    if dz <= 0:
        raise ValueError("dz must be > 0")
    if len(temp) < 2:
        return []
    out = [0.0] * (len(temp) - 1)
    for i in range(len(out)):
        out[i] = -k * (temp[i + 1] - temp[i]) / dz
    return out


def divergence(flux: list[float], dz: float) -> list[float]:
    if dz <= 0:
        raise ValueError("dz must be > 0")
    if len(flux) < 2:
        return []
    out = [0.0] * (len(flux) - 1)
    for i in range(len(out)):
        out[i] = (flux[i + 1] - flux[i]) / dz
    return out
