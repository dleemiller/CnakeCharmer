from __future__ import annotations


def weighted_mse(y: list[float], w: list[float]) -> float:
    sw = sum(w)
    if sw == 0.0:
        return 0.0
    mu = sum(yi * wi for yi, wi in zip(y, w, strict=False)) / sw
    return sum(wi * (yi - mu) * (yi - mu) for yi, wi in zip(y, w, strict=False)) / sw


def best_threshold(
    feature: list[float], target: list[float], weights: list[float]
) -> tuple[float, float]:
    idx = list(range(len(feature)))
    idx.sort(key=lambda i: feature[i])
    best_thr = feature[idx[0]] if idx else 0.0
    best_loss = float("inf")

    for k in range(1, len(idx)):
        i0 = idx[k - 1]
        i1 = idx[k]
        thr = 0.5 * (feature[i0] + feature[i1])
        yl: list[float] = []
        wl: list[float] = []
        yr: list[float] = []
        wr: list[float] = []
        for i in idx:
            if feature[i] <= thr:
                yl.append(target[i])
                wl.append(weights[i])
            else:
                yr.append(target[i])
                wr.append(weights[i])
        loss = weighted_mse(yl, wl) + weighted_mse(yr, wr)
        if loss < best_loss:
            best_loss = loss
            best_thr = thr
    return best_thr, best_loss
