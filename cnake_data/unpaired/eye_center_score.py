def eye_center_score(inv_eye, gx, gy):
    """Compute center-likelihood scores for an eye patch.

    Args:
        inv_eye: 2D list of inverse eye intensities (higher favors likely center).
        gx: 2D list of x-gradients.
        gy: 2D list of y-gradients.

    Returns:
        2D list of accumulated center scores.
    """
    h = len(inv_eye)
    w = len(inv_eye[0]) if h else 0
    out = [[0.0 for _ in range(w)] for _ in range(h)]

    for y in range(h):
        for x in range(w):
            g_x = gx[y][x]
            g_y = gy[y][x]
            if g_x == 0.0 and g_y == 0.0:
                continue
            _accumulate_center_votes(x, y, inv_eye, g_x, g_y, out)

    return out


def _accumulate_center_votes(x, y, inv_eye, g_x, g_y, out):
    h = len(inv_eye)
    w = len(inv_eye[0]) if h else 0

    for cy in range(h):
        for cx in range(w):
            if cx == x and cy == y:
                continue

            dx = x - cx
            dy = y - cy
            norm = (dx * dx + dy * dy) ** 0.5
            if norm != 0.0:
                dx /= norm
                dy /= norm
            else:
                dx = 0.0
                dy = 0.0

            dot = dx * g_x + dy * g_y
            if dot < 0.0:
                dot = 0.0

            out[cy][cx] += (dot * dot) * inv_eye[cy][cx]
