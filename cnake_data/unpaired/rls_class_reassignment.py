"""Local-search class reassignment helpers for matrix-factor style objectives."""

from __future__ import annotations


def claim_all_points(
    labels: list[list[float]],
    class_counts: list[int],
    class_vec: list[int],
    d_vt_y: list[list[float]],
    sqrt_rx2: list[list[float]],
    new_class: int,
) -> None:
    size = len(class_vec)
    rank_r = len(d_vt_y)
    for i in range(size):
        old_class = class_vec[i]
        if old_class == new_class:
            continue

        labels[i][old_class] = -1.0
        labels[i][new_class] = 1.0
        class_vec[i] = new_class
        class_counts[old_class] -= 1
        class_counts[new_class] += 1

        for j in range(rank_r):
            d_vt_y[j][new_class] += sqrt_rx2[i][j]
            d_vt_y[j][old_class] -= sqrt_rx2[i][j]


def cyclic_descent(
    labels: list[list[float]],
    class_counts: list[int],
    class_vec: list[int],
    d_vt_y: list[list[float]],
    sqrt_rx2: list[list[float]],
    global_size: int,
    label_count: int,
    max_balance_change: int,
    classcount_delta: list[int],
) -> int:
    size = len(class_vec)
    rank_r = len(d_vt_y)
    fitvec = [float(global_size)] * label_count

    for cand_class in range(label_count):
        v = float(global_size)
        for j in range(rank_r):
            v -= d_vt_y[j][cand_class] * d_vt_y[j][cand_class]
        fitvec[cand_class] = v

    changed = True
    change_count = 0

    while changed:
        changed = False
        for i in range(size):
            old_class = class_vec[i]
            new_class = old_class

            for cand_class in range(label_count):
                if cand_class == old_class:
                    continue

                temp_old = float(global_size)
                temp_new = float(global_size)
                for j in range(rank_r):
                    t_old = d_vt_y[j][old_class] - sqrt_rx2[i][j]
                    t_new = d_vt_y[j][cand_class] + sqrt_rx2[i][j]
                    temp_old -= t_old * t_old
                    temp_new -= t_new * t_new

                if temp_old + temp_new < fitvec[old_class] + fitvec[cand_class]:
                    labels[i][old_class] = -1.0
                    labels[i][cand_class] = 1.0
                    class_vec[i] = cand_class
                    class_counts[old_class] -= 1
                    class_counts[cand_class] += 1

                    for j in range(rank_r):
                        d_vt_y[j][cand_class] += sqrt_rx2[i][j]
                        d_vt_y[j][old_class] -= sqrt_rx2[i][j]

                    fitvec[old_class] = temp_old
                    fitvec[cand_class] = temp_new
                    changed = True
                    new_class = cand_class

            if changed:
                change_count += 1
                classcount_delta[old_class] -= 1
                classcount_delta[new_class] += 1

                if (
                    abs(classcount_delta[old_class]) >= max_balance_change
                    or abs(classcount_delta[new_class]) >= max_balance_change
                ):
                    changed = False
                    break

    return change_count
