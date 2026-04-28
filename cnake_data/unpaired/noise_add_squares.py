import random

import numpy as np


def add_squares(image, vmax=100.0, vmin=0.0, n_squares=100):
    h, w = image.shape
    x0_arr = np.random.randint(low=0, high=max(w - 1, 1), size=n_squares, dtype=np.int32)
    x1_arr = np.random.randint(low=0, high=max(w - 1, 1), size=n_squares, dtype=np.int32)
    y0_arr = np.random.randint(low=0, high=max(h - 1, 1), size=n_squares, dtype=np.int32)
    y1_arr = np.random.randint(low=0, high=max(h - 1, 1), size=n_squares, dtype=np.int32)

    for n in range(n_squares):
        v = random.random() * (vmax - vmin) + vmin
        x0 = min(int(x0_arr[n]), int(x1_arr[n]))
        x1 = max(int(x0_arr[n]), int(x1_arr[n]))
        y0 = min(int(y0_arr[n]), int(y1_arr[n]))
        y1 = max(int(y0_arr[n]), int(y1_arr[n]))
        image[y0:y1, x0:x1] += v

    return image


def get_squares(w, h, vmax=100.0, vmin=0.0, n_squares=100):
    image = np.zeros((h, w), dtype=np.float32)
    return add_squares(image, vmax=vmax, vmin=vmin, n_squares=n_squares)
