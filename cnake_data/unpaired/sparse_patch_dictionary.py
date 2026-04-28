import math


def normalize_columns(matrix):
    """Normalize each column of matrix in place to unit L2 norm."""
    m = len(matrix)
    n = len(matrix[0]) if m else 0

    for col in range(n):
        acc = 0.0
        for row in range(m):
            acc += matrix[row][col] * matrix[row][col]
        acc = math.sqrt(acc)
        if acc == 0.0:
            continue
        for row in range(m):
            matrix[row][col] /= acc


def construct_dct_dictionary(m, n):
    """Construct a DCT-like dictionary with shape m x n."""
    d = [[0.0 for _ in range(n)] for _ in range(m)]
    if n == 0:
        return d

    c0 = n ** (-0.5)
    for row in range(m):
        d[row][0] = c0

    scale = (2.0 / n) ** 0.5
    for col in range(1, n):
        for row in range(m):
            d[row][col] = scale * math.cos(math.pi / n * (row + 0.5) * col)

    normalize_columns(d)
    return d


def matching_pursuit(x, d, iter_max=-1, global_eps=1e-6):
    """Basic matching pursuit sparse approximation."""
    m = len(d)
    n = len(d[0]) if m else 0
    if iter_max < 0:
        iter_max = n

    r = list(x)
    s = [0.0 for _ in range(n)]

    for _ in range(iter_max):
        w_max = 0.0
        w_argmax = 0

        for col in range(n):
            acc = 0.0
            for row in range(m):
                acc += d[row][col] * r[row]
            if abs(acc) > abs(w_max):
                w_max = acc
                w_argmax = col

        s[w_argmax] = w_max

        acc = 0.0
        for row in range(m):
            r[row] -= w_max * d[row][w_argmax]
            acc += r[row] * r[row]
        if math.sqrt(acc) < global_eps:
            break

    return s, r


def reconstruct_image(height, width, block_size, sparse_dict, sparse_mat):
    """Reconstruct an image from sliding-window sparse coefficients."""
    fused = [[0.0 for _ in range(width)] for _ in range(height)]

    m = len(sparse_dict)
    n_blocks = len(sparse_mat[0]) if sparse_mat else 0

    # fused_recon has shape (m, n_blocks)
    fused_recon = [[0.0 for _ in range(n_blocks)] for _ in range(m)]
    for i in range(m):
        for j in range(n_blocks):
            acc = 0.0
            for k in range(len(sparse_dict[0])):
                acc += sparse_dict[i][k] * sparse_mat[k][j]
            fused_recon[i][j] = acc

    i = 0
    for row in range(height - block_size + 1):
        for col in range(width - block_size + 1):
            for row_ in range(block_size):
                for col_ in range(block_size):
                    fused[row + row_][col + col_] += fused_recon[row_ * block_size + col_][i]
            i += 1

    return fused
