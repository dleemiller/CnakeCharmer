"""Integer-grid crossing extraction from sampled paths."""

from __future__ import annotations

import math

import numpy as np


def crossings(x, t, scale, origin):
    x = np.asarray(x, dtype=float)
    t = np.asarray(t, dtype=float)
    n_samples = x.shape[0]

    size = np.empty(n_samples - 1, dtype=int)
    first = np.empty(n_samples - 1, dtype=float)

    prev_last = np.nan
    total = 0
    xs1 = (x[0] - origin) / scale

    for i in range(n_samples - 1):
        xs0, xs1 = xs1, (x[i + 1] - origin) / scale
        direction, size_ = 0.0, 0

        if x[i] < x[i + 1]:
            first_, last_ = math.ceil(xs0), math.floor(xs1)
            direction = +1.0
        elif x[i] > x[i + 1]:
            first_, last_ = math.floor(xs0), math.ceil(xs1)
            direction = -1.0
        else:
            first_, last_ = np.nan, np.nan

        if direction != 0.0:
            size_ = int(abs(last_ + direction - first_))
            if size_ > 0 and prev_last == first_:
                first_ += direction
                size_ -= 1
            if size_ > 0:
                prev_last = last_

        first[i], size[i] = first_, size_
        total += size_

    xi = np.empty(total, dtype=float)
    ti = np.empty(total, dtype=float)

    j = 0
    for i in range(n_samples - 1):
        size_ = size[i]
        if size_ > 0:
            x_slope = +scale if x[i + 1] > x[i] else -scale
            t_slope = (t[i + 1] - t[i]) / (x[i + 1] - x[i])
            first_level = first[i] * scale + origin
            for k in range(size_):
                xi[j] = first_level + x_slope * k
                ti[j] = t[i] + t_slope * (xi[j] - x[i])
                j += 1

    return xi, ti
