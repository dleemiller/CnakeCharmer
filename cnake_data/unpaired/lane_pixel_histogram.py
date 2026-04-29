from __future__ import annotations


def column_histogram(binary_img: list[list[int]]) -> list[int]:
    h = len(binary_img)
    if h == 0:
        return []
    w = len(binary_img[0])
    hist = [0] * w
    for y in range(h):
        row = binary_img[y]
        for x in range(w):
            if row[x]:
                hist[x] += 1
    return hist


def peak_columns(hist: list[int], k: int) -> list[int]:
    idx = list(range(len(hist)))
    idx.sort(key=lambda i: hist[i], reverse=True)
    return idx[: max(0, k)]
