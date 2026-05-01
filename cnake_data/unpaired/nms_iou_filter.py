from __future__ import annotations


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def nms(
    boxes: list[tuple[float, float, float, float]], scores: list[float], thr: float
) -> list[int]:
    idx = list(range(len(boxes)))
    idx.sort(key=lambda i: scores[i], reverse=True)
    keep: list[int] = []
    while idx:
        i = idx.pop(0)
        keep.append(i)
        nxt: list[int] = []
        bi = boxes[i]
        for j in idx:
            if iou(bi, boxes[j]) <= thr:
                nxt.append(j)
        idx = nxt
    return keep
