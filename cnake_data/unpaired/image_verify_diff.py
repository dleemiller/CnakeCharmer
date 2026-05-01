"""Pixel-wise image comparison and red-channel difference map."""

from __future__ import annotations

import numpy as np


def verify_image(ref: np.ndarray, test: np.ndarray, tolerance: float, cnt: int) -> np.ndarray:
    h, w = ref.shape[:2]
    cmp_img_data = np.zeros((h, w, 3), dtype=np.uint8)

    pix_sum = 0
    err_sum = 0.0
    exact_match = True

    for y in range(h):
        for x in range(w):
            pix_sum += 3
            dr = abs(int(ref[y, x, 0]) - int(test[y, x, 0]))
            dg = abs(int(ref[y, x, 1]) - int(test[y, x, 1]))
            db = abs(int(ref[y, x, 2]) - int(test[y, x, 2]))
            amt_dif = float(dr + dg + db)
            if amt_dif > 0:
                exact_match = False
                err_sum += amt_dif
                cmp_img_data[y, x, 2] = min(255, int(amt_dif / 1.5))

    avg_err = err_sum / max(1, pix_sum)
    print(f"Image {cnt}")
    if exact_match:
        print("Image is exact match. Zero percent tolerance.")
    elif (avg_err / 255.0) < tolerance:
        print(f"Percent tolerance given: {tolerance * 100:.3f}%")
        print(f"Image passed w/ tolerance: {(avg_err / 255.0) * 100:.3f}%")
    else:
        print(f"Percentage tolerance given: {tolerance * 100:.3f}%")
        print(f"Percentage of failure: {(avg_err / 255.0) * 100:.3f}%")
    print("----------------------------------------")
    return cmp_img_data
