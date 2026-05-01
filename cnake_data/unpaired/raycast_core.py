"""Basic raycasting setup and trig table precomputation."""

from __future__ import annotations

import math


def init(width, height, fov, wall_array, floor_ceil_distances):
    map_size_x = len(wall_array)
    map_size_y = len(wall_array[0]) if map_size_x else 0

    sin_table = [0.0] * 3600
    cos_table = [0.0] * 3600
    tan_table = [0.0] * 3600

    angle = 0.0
    for i in range(3600):
        rad = math.radians(angle) + 0.0001
        sin_table[i] = math.sin(rad)
        cos_table[i] = math.cos(rad)
        tan_table[i] = math.tan(rad)
        angle += 0.1

    ray_increment = float(fov) / float(width)
    dist_to_plane = (width / 2.0) / tan_table[(fov // 2) * 10]

    wall_heights = [0] * width
    wall_textures = [0] * width
    wall_tex_offsets = [0] * width

    return {
        "width": width,
        "height": height,
        "fov": fov,
        "map_size_x": map_size_x,
        "map_size_y": map_size_y,
        "walls": wall_array,
        "floor_distances": list(floor_ceil_distances),
        "ray_increment": ray_increment,
        "dist_to_plane": dist_to_plane,
        "sin_table": sin_table,
        "cos_table": cos_table,
        "tan_table": tan_table,
        "wall_heights": wall_heights,
        "wall_textures": wall_textures,
        "wall_tex_offsets": wall_tex_offsets,
    }
