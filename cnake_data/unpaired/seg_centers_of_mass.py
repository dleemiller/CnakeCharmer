"""3D segmentation centroid accumulation."""

from __future__ import annotations


def centers_of_mass(seg, offset=(0, 0, 0)):
    centers = {}
    counts = {}
    sz = len(seg)
    sy = len(seg[0]) if sz else 0
    sx = len(seg[0][0]) if sy else 0

    for z in range(sz):
        for y in range(sy):
            for x in range(sx):
                segid = seg[z][y][x]
                if segid == 0:
                    continue
                if segid not in centers:
                    centers[segid] = [0.0, 0.0, 0.0]
                    counts[segid] = 0
                centers[segid][0] += z
                centers[segid][1] += y
                centers[segid][2] += x
                counts[segid] += 1

    out = {}
    for segid, acc in centers.items():
        c = counts[segid]
        out[segid] = [
            acc[0] / c + offset[0],
            acc[1] / c + offset[1],
            acc[2] / c + offset[2],
        ]
    return out
