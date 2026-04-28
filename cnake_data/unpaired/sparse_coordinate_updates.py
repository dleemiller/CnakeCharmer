def locally_greedy_coordinate_selection(z_data, z_rows, t_start, t_end, tol):
    """Select largest-magnitude coordinate update in [t_start, t_end)."""
    k0 = -1
    t0 = -1
    dz = 0.0
    adz = tol

    for k, (z_k, z_k_row) in enumerate(zip(z_data, z_rows, strict=False)):
        for zk, t in zip(z_k, z_k_row, strict=False):
            if t < t_start:
                continue
            if t >= t_end:
                break
            azk = abs(zk)
            if azk > adz:
                k0 = k
                t0 = t
                adz = azk
                dz = zk

    return k0, t0, dz


def update_dz_opt(z_data, z_rows, beta, dz_opt, norm_d, reg, t_start, t_end):
    """Update per-coordinate descent deltas over a time segment."""
    for k, (z_k, z_k_row) in enumerate(zip(z_data, z_rows, strict=False)):
        norm_dk = norm_d[k]
        current_t = t_start

        for tk, zk in zip(z_k_row, z_k, strict=False):
            if tk < t_start:
                continue
            if tk >= t_end:
                break

            for t in range(current_t, tk):
                tmp = max(-beta[k][t] - reg, 0.0) / norm_dk
                dz_opt[k][t] = tmp

            tmp = max(-beta[k][tk] - reg, 0.0) / norm_dk
            dz_opt[k][tk] = tmp - zk
            current_t = tk + 1

        for t in range(current_t, t_end):
            tmp = max(-beta[k][t] - reg, 0.0) / norm_dk
            dz_opt[k][t] = tmp

    return dz_opt


def subtract_sparse_codes_from_beta(beta, z_data, z_rows, norm_dk):
    """Apply beta[k][t] -= z[k,t] * norm_dk[k] for sparse code rows."""
    for k, (z_k, z_k_row) in enumerate(zip(z_data, z_rows, strict=False)):
        for t, z in zip(z_k_row, z_k, strict=False):
            beta[k][t] -= z * norm_dk[k]
    return beta
