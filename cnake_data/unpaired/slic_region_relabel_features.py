from __future__ import annotations


def create_label_prob_image(
    slic_reg: list[list[list[int]]], label_prob: list[float]
) -> list[list[list[float]]]:
    x = len(slic_reg)
    y = len(slic_reg[0]) if x else 0
    z = len(slic_reg[0][0]) if y else 0
    out = [[[0.0 for _ in range(z)] for _ in range(y)] for _ in range(x)]
    for xx in range(x):
        for yy in range(y):
            for zz in range(z):
                out[xx][yy][zz] = label_prob[slic_reg[xx][yy][zz]]
    return out


def group_labeled_voxels(
    slic_reg: list[list[list[int]]],
    keep_labels: set[int],
    img_feat: list[list[list[list[float]]]],
) -> tuple[list[tuple[int, int, int]], list[list[float]]]:
    coords: list[tuple[int, int, int]] = []
    feats: list[list[float]] = []
    x = len(slic_reg)
    y = len(slic_reg[0]) if x else 0
    z = len(slic_reg[0][0]) if y else 0
    for xx in range(x):
        for yy in range(y):
            for zz in range(z):
                if slic_reg[xx][yy][zz] in keep_labels:
                    coords.append((xx, yy, zz))
                    feats.append(list(img_feat[xx][yy][zz]))
    return coords, feats
