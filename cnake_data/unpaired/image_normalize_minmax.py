from __future__ import annotations


def get_min_max(image: list[list[float]]) -> tuple[float, float]:
    amin = image[0][0]
    amax = image[0][0]
    for row in image:
        for v in row:
            if v > amax:
                amax = v
            elif v < amin:
                amin = v
    return amin, amax


def normalize_image_uint8(image: list[list[float]]) -> tuple[list[list[int]], tuple[float, float]]:
    amin, amax = get_min_max(image)
    if amax == amin:
        return [[0 for _ in row] for row in image], (amin, amax)
    factor = 255.0 / (amax - amin)
    out: list[list[int]] = []
    for row in image:
        out.append([int((v - amin) * factor) & 0xFF for v in row])
    return out, (amin, amax)
