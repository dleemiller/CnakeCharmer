"""Bounding-box overlap metrics (IoU / IoM)."""

from __future__ import annotations

import numpy as np


def bbox_overlaps(boxes: np.ndarray, query_boxes: np.ndarray, method: int = 0) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=float)
    query_boxes = np.asarray(query_boxes, dtype=float)
    n = boxes.shape[0]
    k = query_boxes.shape[0]
    overlaps = np.zeros((n, k), dtype=float)

    for j in range(k):
        box_area = (query_boxes[j, 2] - query_boxes[j, 0] + 1) * (
            query_boxes[j, 3] - query_boxes[j, 1] + 1
        )
        for i in range(n):
            iw = min(boxes[i, 2], query_boxes[j, 2]) - max(boxes[i, 0], query_boxes[j, 0]) + 1
            if iw <= 0:
                continue
            ih = min(boxes[i, 3], query_boxes[j, 3]) - max(boxes[i, 1], query_boxes[j, 1]) + 1
            if ih <= 0:
                continue
            boxes_area = (boxes[i, 2] - boxes[i, 0] + 1) * (boxes[i, 3] - boxes[i, 1] + 1)
            ua = boxes_area + box_area - iw * ih if method == 0 else min(boxes_area, box_area)
            overlaps[i, j] = iw * ih / float(ua)

    return overlaps
