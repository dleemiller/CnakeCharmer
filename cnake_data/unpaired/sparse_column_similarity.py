import math


def cosine_common_csc(indices, indptr, data, ncols):
    """Cosine similarity using only rows shared by each CSC column pair."""
    result = [[0.0 for _ in range(ncols)] for _ in range(ncols)]
    common = [[0 for _ in range(ncols)] for _ in range(ncols)]

    for i in range(ncols):
        i0 = indptr[i]
        i1 = indptr[i + 1]
        n_i = i1 - i0

        for j in range(i + 1, ncols):
            j0 = indptr[j]
            j1 = indptr[j + 1]
            n_j = j1 - j0

            ii = 0
            jj = 0
            n_common = 0
            ij_sum = 0.0
            ii_sum = 0.0
            jj_sum = 0.0

            while ii < n_i and jj < n_j:
                ri = indices[i0 + ii]
                rj = indices[j0 + jj]
                if ri < rj:
                    ii += 1
                elif ri > rj:
                    jj += 1
                else:
                    x_i = data[i0 + ii]
                    x_j = data[j0 + jj]
                    ij_sum += x_i * x_j
                    ii_sum += x_i * x_i
                    jj_sum += x_j * x_j
                    ii += 1
                    jj += 1
                    n_common += 1

            if n_common > 0:
                denom = math.sqrt(ii_sum * jj_sum)
                c = (ij_sum / denom) if denom > 0.0 else 0.0
                result[i][j] = c
                result[j][i] = c
                common[i][j] = n_common
                common[j][i] = n_common

    return result, common


def pearson_common_csc(indices, indptr, data, ncols):
    """Pearson correlation over rows shared by each CSC column pair."""
    result = [[0.0 for _ in range(ncols)] for _ in range(ncols)]
    common = [[0 for _ in range(ncols)] for _ in range(ncols)]

    for i in range(ncols):
        i0 = indptr[i]
        i1 = indptr[i + 1]
        n_i = i1 - i0

        for j in range(i, ncols):
            j0 = indptr[j]
            j1 = indptr[j + 1]
            n_j = j1 - j0

            ii = 0
            jj = 0
            n_common = 0
            i_sum = 0.0
            j_sum = 0.0
            ij_sum = 0.0
            ii_sum = 0.0
            jj_sum = 0.0

            while ii < n_i and jj < n_j:
                ri = indices[i0 + ii]
                rj = indices[j0 + jj]
                if ri < rj:
                    ii += 1
                elif ri > rj:
                    jj += 1
                else:
                    x_i = data[i0 + ii]
                    x_j = data[j0 + jj]
                    i_sum += x_i
                    j_sum += x_j
                    ij_sum += x_i * x_j
                    ii_sum += x_i * x_i
                    jj_sum += x_j * x_j
                    ii += 1
                    jj += 1
                    n_common += 1

            if n_common > 0:
                num = n_common * ij_sum - i_sum * j_sum
                den = math.sqrt(
                    (n_common * ii_sum - i_sum * i_sum) * (n_common * jj_sum - j_sum * j_sum)
                )
                c = (num / den) if den > 0.0 else 0.0
                result[i][j] = c
                result[j][i] = c
                common[i][j] = n_common
                common[j][i] = n_common

    return result, common
