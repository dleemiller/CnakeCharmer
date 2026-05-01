"""Euclidean distance-based region growing on 3D grids."""

from __future__ import annotations

from collections import deque


def calculate_euclidean(object_area, base_object, neighbourhood, distance):
    z_size = len(object_area)
    y_size = len(object_area[0])
    x_size = len(object_area[0][0])

    consumed = [
        [[base_object[z][y][x] for x in range(x_size)] for y in range(y_size)]
        for z in range(z_size)
    ]
    inf = float("inf")
    result = [[[inf for _ in range(x_size)] for _ in range(y_size)] for _ in range(z_size)]

    q = deque()
    for z in range(z_size):
        for y in range(y_size):
            for x in range(x_size):
                if base_object[z][y][x] != 0:
                    result[z][y][x] = 0.0
                    q.append((z, y, x))

    while q:
        pz, py, px = q.popleft()
        consumed[pz][py][px] = 0
        for i, (dz, dy, dx) in enumerate(neighbourhood):
            z = pz + dz
            y = py + dy
            x = px + dx
            if x < 0 or y < 0 or z < 0 or x >= x_size or y >= y_size or z >= z_size:
                continue
            if object_area[z][y][x] == 0:
                continue
            cand = result[pz][py][px] + distance[i]
            if result[z][y][x] > cand:
                result[z][y][x] = cand
                if consumed[z][y][x] == 0:
                    consumed[z][y][x] = 1
                    q.append((z, y, x))
    return result
