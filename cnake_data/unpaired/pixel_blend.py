def pixel_blend(n):
    """Blend two rows of n RGBA pixels using alpha compositing.

    Returns (sum_red, sum_green, sum_blue).
    """
    src_r = [0] * n
    src_g = [0] * n
    src_b = [0] * n
    src_a = [0] * n
    dst_r = [0] * n
    dst_g = [0] * n
    dst_b = [0] * n

    for i in range(n):
        src_r[i] = (i * 7 + 3) % 256
        src_g[i] = (i * 13 + 7) % 256
        src_b[i] = (i * 19 + 11) % 256
        src_a[i] = (i * 5 + 1) % 256
        dst_r[i] = (i * 11 + 5) % 256
        dst_g[i] = (i * 17 + 9) % 256
        dst_b[i] = (i * 23 + 13) % 256

    sum_r = 0
    sum_g = 0
    sum_b = 0
    for i in range(n):
        alpha = src_a[i] / 255.0
        out_r = int(src_r[i] * alpha + dst_r[i] * (1.0 - alpha))
        out_g = int(src_g[i] * alpha + dst_g[i] * (1.0 - alpha))
        out_b = int(src_b[i] * alpha + dst_b[i] * (1.0 - alpha))
        sum_r += out_r
        sum_g += out_g
        sum_b += out_b

    return (sum_r, sum_g, sum_b)
