from __future__ import annotations


def convolve_mask_aware(
    arr: list[list[int]], mask: list[list[int]], threshold: float = 0.0
) -> list[list[float]]:
    h = len(mask)
    w = len(mask[0]) if h else 0
    out = [[0.0 for _ in range(w)] for _ in range(h)]
    for j in range(h):
        for i in range(w):
            if mask[j][i] == 0:
                continue
            s = float(arr[j][i])
            x = 0.0
            neighbors = 0
            for dj, di in [(-1, 0), (1, 0), (0, 1), (0, -1)]:
                nj = j + dj
                ni = i + di
                if 0 <= nj < h and 0 <= ni < w:
                    if mask[nj][ni] != 0:
                        neighbors += 1
                        x += -float(arr[nj][ni])
                    else:
                        d = abs(s - float(arr[nj][ni]))
                        if d > threshold:
                            neighbors += 1
                            x += -float(arr[nj][ni])
            out[j][i] = x + neighbors * float(arr[j][i])
    return out
