import random


def add_noise(img_array, p, img_h, img_w):
    """Salt-and-pepper noise on HxWx3 uint-like image array."""
    noise = int(p * img_w * img_h)
    for _ in range(noise):
        x = random.randint(0, img_h - 1)
        y = random.randint(0, img_w - 1)
        val = 10 if random.randint(0, 1) == 0 else 255
        for ch in range(3):
            img_array[x][y][ch] = val
    return img_array


def floor_int(n):
    return int(n) if n >= 0.0 else int(n - 1)


def resize_bilinear(img_array, out_h, out_w):
    """Simple bilinear resize for HxWxC arrays."""
    in_h = len(img_array)
    in_w = len(img_array[0]) if in_h else 0
    channels = len(img_array[0][0]) if in_w else 0

    if in_h == 0 or in_w == 0:
        return []

    dst = [[[0 for _ in range(channels)] for _ in range(out_w)] for _ in range(out_h)]
    scale_x = in_w / float(out_w)
    scale_y = in_h / float(out_h)

    for dy in range(out_h):
        for dx in range(out_w):
            src_x = (dx + 0.5) * scale_x - 0.5
            src_y = (dy + 0.5) * scale_y - 0.5

            x0 = floor_int(src_x)
            y0 = floor_int(src_y)
            x1 = min(max(x0 + 1, 0), in_w - 1)
            y1 = min(max(y0 + 1, 0), in_h - 1)
            x0 = min(max(x0, 0), in_w - 1)
            y0 = min(max(y0, 0), in_h - 1)

            wx = src_x - x0
            wy = src_y - y0

            for k in range(channels):
                v00 = img_array[y0][x0][k]
                v01 = img_array[y0][x1][k]
                v10 = img_array[y1][x0][k]
                v11 = img_array[y1][x1][k]
                v0 = (1.0 - wx) * v00 + wx * v01
                v1 = (1.0 - wx) * v10 + wx * v11
                dst[dy][dx][k] = int((1.0 - wy) * v0 + wy * v1)

    return dst
