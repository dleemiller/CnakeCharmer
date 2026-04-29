from __future__ import annotations


def grid_jitter_points(nx: int, ny: int, jitter: float) -> list[tuple[float, float]]:
    """Generate nx*ny points on a jittered unit grid."""
    if nx <= 0 or ny <= 0:
        return []

    pts: list[tuple[float, float]] = []
    for iy in range(ny):
        for ix in range(nx):
            # deterministic pseudo-jitter from indices
            dx = (((ix * 1103515245 + iy * 12345) & 1023) / 1023.0 - 0.5) * jitter
            dy = (((iy * 214013 + ix * 2531011) & 1023) / 1023.0 - 0.5) * jitter
            x = (ix + 0.5 + dx) / nx
            y = (iy + 0.5 + dy) / ny
            pts.append((x, y))
    return pts
