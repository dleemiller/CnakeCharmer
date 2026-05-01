from __future__ import annotations


def cluster_by_intervals(
    starts: list[int], ends: list[int], ids: list[int], slack: int = 0
) -> list[int]:
    if not starts or len(starts) != len(ends) or len(starts) != len(ids):
        return []
    max_end = ends[0]
    last_id = ids[0]
    n_clusters = 1
    out = [-1] * len(starts)
    for i in range(len(starts)):
        cur_id = ids[i]
        if (starts[i] - slack) > max_end or cur_id != last_id:
            n_clusters += 1
            max_end = ends[i]
        else:
            if ends[i] > max_end:
                max_end = ends[i]
        out[i] = n_clusters
        last_id = cur_id
    return out
