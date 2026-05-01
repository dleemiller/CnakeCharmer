from __future__ import annotations


def conv2d_clamped(image: list[list[float]], kernel: list[list[float]]) -> list[list[float]]:
    n_rows = len(image)
    n_cols = len(image[0]) if n_rows else 0
    k_rows = len(kernel)
    k_cols = len(kernel[0]) if k_rows else 0
    cr = (k_rows - 1) // 2
    cc = (k_cols - 1) // 2
    out = [[0.0 for _ in range(n_cols)] for _ in range(n_rows)]
    for r in range(n_rows):
        for c in range(n_cols):
            acc = 0.0
            for kr in range(k_rows):
                for kc in range(k_cols):
                    rr = r + (kr - cr)
                    cc2 = c + (kc - cc)
                    if rr < 0:
                        rr = 0
                    elif rr >= n_rows:
                        rr = n_rows - 1
                    if cc2 < 0:
                        cc2 = 0
                    elif cc2 >= n_cols:
                        cc2 = n_cols - 1
                    acc += kernel[kr][kc] * image[rr][cc2]
            out[r][c] = acc
    return out
