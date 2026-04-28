def step(
    f,
    fp,
    nx,
    ny,
    nxi,
    model_padded2_dt2,
    sources,
    sources_x,
    sources_y,
    num_sources,
    source_len,
    num_steps,
    fd_coeff,
):
    for step_i in range(num_steps):
        for i in range(8, ny - 8):
            for j in range(8, nxi + 8):
                f_xx = fd_coeff[0] * f[i, j]
                for k in range(1, 9):
                    f_xx += fd_coeff[k] * (f[i, j + k] + f[i, j - k] + f[i + k, j] + f[i - k, j])

                fp[i, j] = model_padded2_dt2[i, j] * f_xx + 2 * f[i, j] - fp[i, j]

        for i in range(num_sources):
            sx = sources_x[i] + 8
            sy = sources_y[i] + 8
            fp[sy, sx] += model_padded2_dt2[sy, sx] * sources[i, step_i]

        f, fp = fp, f

    return f, fp
