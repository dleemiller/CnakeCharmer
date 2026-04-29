"""Convert RGB image to grayscale using min/max midpoint."""

from __future__ import annotations

import numpy as np


def gray(num1, num2, img):
    a = int(num1)
    b = int(num2)
    arr = np.asarray(img).copy()

    for i in range(a):
        for j in range(b):
            red = int(arr[i, j, 2])
            green = int(arr[i, j, 1])
            blue = int(arr[i, j, 0])
            avg = int(min(red, green, blue) // 2 + max(red, green, blue) // 2)
            arr[i, j] = avg
    return arr
