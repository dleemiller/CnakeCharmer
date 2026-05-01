"""Max-pooling forward/backward operations."""

from __future__ import annotations


def max_pool_forward(data, filter_height, filter_width, stride_height, stride_width):
    b = len(data)
    c = len(data[0])
    h = len(data[0][0])
    w = len(data[0][0][0])
    n_h = int(1 + (h - filter_height) / stride_height)
    n_w = int(1 + (w - filter_width) / stride_width)
    out = [[[[0.0 for _ in range(n_w)] for _ in range(n_h)] for _ in range(c)] for _ in range(b)]
    for i in range(b):
        for ch in range(c):
            for yy in range(n_h):
                for xx in range(n_w):
                    vs = yy * stride_height
                    ve = vs + filter_height
                    hs = xx * stride_width
                    he = hs + filter_width
                    m = -1e15
                    for ii in range(vs, ve):
                        for jj in range(hs, he):
                            v = data[i][ch][ii][jj]
                            if v > m:
                                m = v
                    out[i][ch][yy][xx] = m
    return out
