def pool_bc01(imgs, pool_h, pool_w, stride_y, stride_x):
    """Max-pool over BC01 tensor layout.

    Args:
        imgs: shape [n_imgs][n_channels][img_h][img_w]

    Returns:
        poolout, switches
        poolout: [n_imgs][n_channels][out_h][out_w]
        switches: [n_imgs][n_channels][out_h][out_w] -> (img_y, img_x)
    """
    n_imgs = len(imgs)
    n_channels = len(imgs[0]) if n_imgs else 0
    img_h = len(imgs[0][0]) if n_channels else 0
    img_w = len(imgs[0][0][0]) if img_h else 0

    out_h = img_h // stride_y
    out_w = img_w // stride_x

    pool_h_top = pool_h // 2 - 1 + pool_h % 2
    pool_h_bottom = pool_h // 2 + 1
    pool_w_left = pool_w // 2 - 1 + pool_w % 2
    pool_w_right = pool_w // 2 + 1

    poolout = [
        [[[0.0 for _ in range(out_w)] for _ in range(out_h)] for _ in range(n_channels)]
        for _ in range(n_imgs)
    ]
    switches = [
        [[[(-1, -1) for _ in range(out_w)] for _ in range(out_h)] for _ in range(n_channels)]
        for _ in range(n_imgs)
    ]

    for i in range(n_imgs):
        for c in range(n_channels):
            for y_out in range(out_h):
                y = y_out * stride_y
                y_min = max(y - pool_h_top, 0)
                y_max = min(y + pool_h_bottom, img_h)

                for x_out in range(out_w):
                    x = x_out * stride_x
                    x_min = max(x - pool_w_left, 0)
                    x_max = min(x + pool_w_right, img_w)

                    best = float("-inf")
                    best_pos = (0, 0)
                    for img_y in range(y_min, y_max):
                        for img_x in range(x_min, x_max):
                            v = imgs[i][c][img_y][img_x]
                            if v > best:
                                best = v
                                best_pos = (img_y, img_x)

                    poolout[i][c][y_out][x_out] = best
                    switches[i][c][y_out][x_out] = best_pos

    return poolout, switches


def bprop_pool_bc01(poolout_grad, switches, img_h, img_w):
    """Backpropagate max-pool gradient using argmax switches."""
    n_imgs = len(poolout_grad)
    n_channels = len(poolout_grad[0]) if n_imgs else 0
    out_h = len(poolout_grad[0][0]) if n_channels else 0
    out_w = len(poolout_grad[0][0][0]) if out_h else 0

    imgs_grad = [
        [[[0.0 for _ in range(img_w)] for _ in range(img_h)] for _ in range(n_channels)]
        for _ in range(n_imgs)
    ]

    for i in range(n_imgs):
        for c in range(n_channels):
            for y in range(out_h):
                for x in range(out_w):
                    img_y, img_x = switches[i][c][y][x]
                    imgs_grad[i][c][img_y][img_x] = poolout_grad[i][c][y][x]

    return imgs_grad
