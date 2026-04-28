import numpy as np


def points_in_disjoint_regions(point_ids, point_coords, region_ids, region_starts, region_ends):
    pt_n = len(point_ids)
    reg_n = len(region_ids)

    id_pairs = np.empty((pt_n, 2), dtype="long")

    i = 0
    j = 0
    k = 0
    while i < pt_n and j < reg_n:
        pt_coord = point_coords[i]
        if pt_coord < region_starts[j]:
            i += 1
        elif pt_coord > region_ends[j]:
            j += 1
        else:
            id_pairs[k, 0] = point_ids[i]
            id_pairs[k, 1] = region_ids[j]
            k += 1
            i += 1

    return id_pairs[:k]
